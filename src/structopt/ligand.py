"""Ligand helpers and GAFF integration."""

import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

from structopt.config import OptimizationConfig

LOGGER = logging.getLogger(__name__)


def _max_valence(atomic_number: int) -> int:
    return {
        1: 1,
        6: 4,
        7: 3,
        8: 2,
        9: 1,
        15: 5,
        16: 6,
        17: 1,
    }.get(atomic_number, 4)


def _prepare_atoms_for_hydrogen_completion(mol: object) -> None:
    for atom in mol.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetNoImplicit(False)
        atom.UpdatePropertyCache(strict=False)


def _assign_pose_bond_orders_from_distances(mol: object) -> None:
    from rdkit import Chem

    conf = mol.GetConformer()
    valence_usage = {atom.GetIdx(): 0 for atom in mol.GetAtoms()}
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        valence_usage[i] += 1
        valence_usage[j] += 1

    candidates: list[tuple[float, object]] = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)
        if atom_i.GetAtomicNum() != 6 or atom_j.GetAtomicNum() != 6:
            continue
        distance = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
        if distance < 1.45:
            candidates.append((distance, bond))

    for _, bond in sorted(candidates, key=lambda item: item[0]):
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        atom_i = mol.GetAtomWithIdx(i)
        atom_j = mol.GetAtomWithIdx(j)
        if valence_usage[i] + 1 > _max_valence(atom_i.GetAtomicNum()):
            continue
        if valence_usage[j] + 1 > _max_valence(atom_j.GetAtomicNum()):
            continue
        bond.SetBondType(Chem.BondType.DOUBLE)
        valence_usage[i] += 1
        valence_usage[j] += 1


def topology_has_ligand(topology: object, ligand_name: str) -> bool:
    target = ligand_name.upper()
    for residue in topology.residues():
        if residue.name.upper() == target:
            return True
    return False


def _find_ligand_residue(topology: object, ligand_name: str) -> object | None:
    target = ligand_name.upper()
    for residue in topology.residues():
        if residue.name.upper() == target:
            return residue
    return None


def _load_openff_molecules_from_sdf(sdf_path: str) -> list[object]:
    try:
        from openff.toolkit import Molecule
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError(
            "Ligand SDF support requires openff-toolkit and RDKit. Install in your environment."
        ) from exc

    LOGGER.info("Loading ligand molecules from SDF: %s", sdf_path)
    rdkit_molecules = [
        mol for mol in Chem.SDMolSupplier(sdf_path, removeHs=False) if mol is not None
    ]
    molecules = [
        Molecule.from_rdkit(
            mol,
            allow_undefined_stereo=True,
            hydrogens_are_explicit=True,
        )
        for mol in rdkit_molecules
    ]
    if not molecules:
        raise RuntimeError(f"No molecules were read from ligand SDF: {sdf_path}")
    LOGGER.info("Loaded %d ligand molecule(s) from SDF", len(molecules))
    return molecules


def _element_symbol(atom_name: str, periodic_table: object) -> str:
    letters = "".join(ch for ch in atom_name if ch.isalpha())
    if not letters:
        return "C"
    if len(letters) >= 2:
        candidate = letters[0].upper() + letters[1].lower()
        if periodic_table.GetAtomicNumber(candidate) > 0:
            return candidate
    return letters[0].upper()


def _build_rdkit_residue_graph(
    residue: object,
    positions: object,
    *,
    infer_bond_orders: bool,
) -> tuple[object, list[object]]:
    from openmm import unit
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from rdkit.Geometry import Point3D

    periodic_table = Chem.GetPeriodicTable()
    atoms = list(residue.atoms())

    rw_mol = Chem.RWMol()
    conformer = Chem.Conformer(len(atoms))
    for idx, atom in enumerate(atoms):
        coords = positions[atom.index].value_in_unit(unit.angstrom)
        element = (
            atom.element.symbol
            if atom.element is not None
            else _element_symbol(atom.name, periodic_table)
        )
        rw_mol.AddAtom(Chem.Atom(element))
        conformer.SetAtomPosition(
            idx,
            Point3D(float(coords.x), float(coords.y), float(coords.z)),
        )

    mol = rw_mol.GetMol()
    conformer.Set3D(True)
    mol.AddConformer(conformer, assignId=True)

    if infer_bond_orders:
        try:
            rdDetermineBonds.DetermineBonds(mol, charge=0)
            LOGGER.debug("Inferred ligand connectivity and bond orders from input pose")
        except Exception:
            LOGGER.warning(
                "Bond-order inference failed for ligand residue '%s'; "
                "falling back to pose-distance bond-order heuristic",
                residue.name,
            )
            rdDetermineBonds.DetermineConnectivity(mol, useVdw=True)
            _assign_pose_bond_orders_from_distances(mol)
    else:
        rdDetermineBonds.DetermineConnectivity(mol, useVdw=True)

    _prepare_atoms_for_hydrogen_completion(mol)
    return mol, atoms


def _ensure_ligand_bonds(topology: object, positions: object, ligand_name: str) -> None:
    target = ligand_name.upper()
    if not hasattr(topology, "bonds"):
        return

    for residue in topology.residues():
        if residue.name.upper() != target or not hasattr(residue, "atoms"):
            continue

        rd_mol, atoms = _build_rdkit_residue_graph(
            residue,
            positions,
            infer_bond_orders=False,
        )
        if rd_mol.GetNumBonds() == 0:
            LOGGER.warning("No ligand bonds were inferred for residue '%s'", residue.name)
            continue

        local_pairs: set[tuple[int, int]] = set()
        atom_indices = {atom.index for atom in atoms}
        atom_to_local = {atom.index: i for i, atom in enumerate(atoms)}

        for atom1, atom2 in topology.bonds():
            if atom1.index not in atom_indices or atom2.index not in atom_indices:
                continue
            i = atom_to_local[atom1.index]
            j = atom_to_local[atom2.index]
            local_pairs.add((i, j) if i < j else (j, i))

        added = 0
        for bond in rd_mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            pair = (i, j) if i < j else (j, i)
            if pair in local_pairs:
                continue
            topology.addBond(atoms[i], atoms[j])
            added += 1

        LOGGER.info(
            "Ligand bond reconciliation for '%s': added %d missing bonds",
            residue.name,
            added,
        )


def _extract_ligand_sdf_from_topology(
    topology: object,
    positions: object,
    ligand_name: str,
) -> Path:
    try:
        from rdkit import Chem
    except ImportError as exc:
        raise RuntimeError(
            "Auto ligand extraction requires RDKit. Install with: uv sync --extra gaff"
        ) from exc

    ligand_residue = _find_ligand_residue(topology, ligand_name)
    if ligand_residue is None:
        raise RuntimeError(f"Ligand residue '{ligand_name}' was not found in topology")

    LOGGER.info("Extracting ligand '%s' directly from input pose coordinates", ligand_name)
    pose_mol, residue_atoms = _build_rdkit_residue_graph(
        ligand_residue,
        positions,
        infer_bond_orders=True,
    )
    if pose_mol.GetNumAtoms() == 0:
        raise RuntimeError(f"Could not derive ligand '{ligand_name}' from coordinates")

    residue_h_count = sum(
        1
        for atom in residue_atoms
        if atom.element is not None and atom.element.atomic_number == 1
    )
    if residue_h_count == 0:
        LOGGER.info(
            "Ligand '%s' has no hydrogens in input pose; adding hydrogens with "
            "coordinates projected from the provided pose",
            ligand_name,
        )
        pose_mol = Chem.AddHs(pose_mol, addCoords=True)
        try:
            from openmm import Vec3, unit
            from openmm.app import element as elem
        except ImportError as exc:
            raise RuntimeError(
                "OpenMM is required to add inferred ligand hydrogens to the topology."
            ) from exc

        LOGGER.info(
            "Injecting inferred ligand hydrogens into topology to match GAFF molecule definition"
        )
        conformer = pose_mol.GetConformer()
        h_counter = 1
        for atom in pose_mol.GetAtoms():
            if atom.GetAtomicNum() != 1:
                continue
            neighbors = list(atom.GetNeighbors())
            if len(neighbors) != 1:
                continue
            heavy_idx = neighbors[0].GetIdx()
            if heavy_idx >= len(residue_atoms):
                continue
            heavy_atom = residue_atoms[heavy_idx]
            h_atom = topology.addAtom(f"HX{h_counter}", elem.hydrogen, ligand_residue)
            topology.addBond(heavy_atom, h_atom)
            h_pos = conformer.GetAtomPosition(atom.GetIdx())
            positions.append(Vec3(float(h_pos.x), float(h_pos.y), float(h_pos.z)) * unit.angstrom)
            h_counter += 1
        LOGGER.info("Injected %d ligand hydrogens into topology", h_counter - 1)
    else:
        LOGGER.info("Ligand '%s' already has explicit hydrogens in input pose", ligand_name)
    _prepare_atoms_for_hydrogen_completion(pose_mol)
    Chem.SanitizeMol(pose_mol)

    with NamedTemporaryFile(prefix=f"structopt_{ligand_name}_", suffix=".sdf", delete=False) as fh:
        sdf_path = Path(fh.name)

    writer = Chem.SDWriter(str(sdf_path))
    writer.write(pose_mol)
    writer.close()
    LOGGER.info("Wrote auto-extracted ligand SDF: %s", sdf_path)
    return sdf_path


def register_gaff_template(
    forcefield: object,
    config: OptimizationConfig,
    topology: object,
    positions: object | None = None,
) -> bool:
    """Register GAFF template generator when ligand handling is requested.

    Returns True when GAFF registration was attempted, False if ligand is absent.
    """
    if not topology_has_ligand(topology, config.ligand_name):
        LOGGER.info("No ligand named '%s' found in topology; skipping GAFF", config.ligand_name)
        return False

    LOGGER.info("Detected ligand residue '%s' in topology", config.ligand_name)
    if config.ligand_sdf is None:
        if positions is None:
            raise RuntimeError(
                f"Ligand residue '{config.ligand_name}' was found in the structure but no ligand "
                "molecule definition was provided and no coordinates were available for automatic "
                "extraction. Pass --ligand-sdf path/to/ligand.sdf."
            )
        LOGGER.info("No ligand SDF provided; using automatic extraction from the input pose")
        _ensure_ligand_bonds(
            topology=topology,
            positions=positions,
            ligand_name=config.ligand_name,
        )
        config.ligand_sdf = _extract_ligand_sdf_from_topology(
            topology=topology,
            positions=positions,
            ligand_name=config.ligand_name,
        )
    elif positions is not None:
        LOGGER.info("Using user-provided ligand SDF: %s", config.ligand_sdf)
        _ensure_ligand_bonds(
            topology=topology,
            positions=positions,
            ligand_name=config.ligand_name,
        )

    try:
        from openmmforcefields.generators import GAFFTemplateGenerator
    except ImportError as exc:
        raise RuntimeError(
            "GAFF support requires openmmforcefields. Install with: uv sync --extra gaff"
        ) from exc

    molecules = _load_openff_molecules_from_sdf(str(config.ligand_sdf))

    kwargs: dict[str, object] = {
        "forcefield": config.gaff_forcefield,
        "molecules": molecules,
    }

    LOGGER.info("Registering GAFF template generator (%s)", config.gaff_forcefield)
    gaff = GAFFTemplateGenerator(**kwargs)
    forcefield.registerTemplateGenerator(gaff.generator)
    LOGGER.info("GAFF template registration completed")
    return True
