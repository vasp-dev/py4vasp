from py4vasp.data import _util, Viewer3d, Topology, Magnetism
from py4vasp.data._base import DataBase, RefinementDescriptor
from IPython.lib.pretty import pretty
from dataclasses import dataclass
import py4vasp.exceptions as exception
import ase
import numpy as np
import functools


class Structure(DataBase):
    """The structure of the crystal.

    You can use this class to process structural information from the Vasp
    calculation. Typically you want to do this to inspect the converged structure
    after an ionic relaxation.

    Parameters
    ----------
    raw_structure : RawStructure
        Dataclass containing the raw data defining the structure.
    """

    length_moments = 1.5
    "Length in Å how a magnetic moment is displayed relative to the largest moment."
    read = RefinementDescriptor("_to_dict")
    to_dict = RefinementDescriptor("_to_dict")
    plot = RefinementDescriptor("_to_viewer3d")
    to_viewer3d = RefinementDescriptor("_to_viewer3d")
    to_ase = RefinementDescriptor("_to_ase")
    to_poscar = RefinementDescriptor("_to_string")
    __str__ = RefinementDescriptor("_to_string")
    _repr_html_ = RefinementDescriptor("_to_html")
    __len__ = RefinementDescriptor("_length")


def _to_string(raw_struct):
    " Generate a string representing this structure usable as a POSCAR file."
    return _create_repr(raw_struct, format_=_Format())


def _to_html(raw_struct):
    format_ = _Format(
        begin="<table>\n<tr><td>",
        separator="</td><td>",
        row="</td></tr>\n<tr><td>",
        end="</td></tr>\n</table>",
        newline="<br>",
    )
    return _create_repr(raw_struct, format_)


@dataclass
class _Format:
    begin: str = ""
    separator: str = " "
    row: str = "\n"
    end: str = ""
    newline: str = ""


def _create_repr(raw_struct, format_):
    cell = raw_struct.cell.scale * raw_struct.cell.lattice_vectors[:]
    vec_to_string = lambda vec: format_.separator.join(str(v) for v in vec)
    vecs_to_string = lambda vecs: format_.row.join(vec_to_string(v) for v in vecs)
    vecs_to_table = lambda vecs: format_.begin + vecs_to_string(vecs) + format_.end
    return f"""
{pretty(Topology(raw_struct.topology))}{format_.newline}
1.0{format_.newline}
{vecs_to_table(cell)}
{Topology(raw_struct.topology).to_poscar(format_.newline)}{format_.newline}
Direct{format_.newline}
{vecs_to_table(raw_struct.positions)}
    """.strip()


def _to_dict(raw_struct):
    """Read the structual information into a dictionary.

    Returns
    -------
    dict
        Contains the unit cell of the crystal, as well as the position of
        all the atoms in units of the lattice vectors and the elements of
        the atoms.
    """
    moments = _read_magnetic_moments(raw_struct.magnetism)
    return {
        "lattice_vectors": raw_struct.cell.scale * raw_struct.cell.lattice_vectors[:],
        "positions": raw_struct.positions[:],
        "elements": Topology(raw_struct.topology).elements(),
        **({"magnetic_moments": moments} if moments is not None else {}),
    }


def _to_viewer3d(raw_struct, supercell=None):
    """Generate a 3d representation of the structure.

    Parameters
    ----------
    supercell : int or np.ndarray
        If present the structure is replicated the specified number of times
        along each direction.

    Returns
    -------
    Viewer3d
        Visualize the structure as a 3d figure, adding the magnetic momements
        of the atoms if present.
    """
    viewer = Viewer3d.from_structure(Structure(raw_struct), supercell=supercell)
    viewer.show_cell()
    moments = _prepare_magnetic_moments_for_plotting(raw_struct.magnetism)
    if moments is not None:
        viewer.show_arrows_at_atoms(moments)
    return viewer


def _to_ase(raw_struct, supercell=None):
    """Convert the structure to an ase Atoms object.

    Parameters
    ----------
    supercell : int or np.ndarray
        If present the structure is replicated the specified number of times
        along each direction.

    Returns
    -------
    ase.Atoms
        Structural information for ase package.
    """
    data = _to_dict(raw_struct)
    structure = ase.Atoms(
        symbols=data["elements"],
        cell=data["lattice_vectors"],
        scaled_positions=data["positions"],
        pbc=True,
    )
    if "magnetic_moments" in data:
        structure.set_initial_magnetic_moments(data["magnetic_moments"])
    if supercell is not None:
        try:
            structure *= supercell
        except (TypeError, IndexError) as err:
            error_message = (
                "Generating the supercell failed. Please make sure the requested "
                "supercell is either an integer or a list of 3 integers."
            )
            raise exception.IncorrectUsage(error_message) from err
    return structure


def _length(raw_struct):
    return len(raw_struct.positions)


def _read_magnetic_moments(magnetism):
    if magnetism is not None:
        return Magnetism(magnetism).total_moments(-1)
    else:
        return None


def _prepare_magnetic_moments_for_plotting(magnetism):
    moments = _read_magnetic_moments(magnetism)
    moments = _convert_to_moment_to_3d_vector(moments)
    max_length_moments = _max_length_moments(moments)
    if max_length_moments > 1e-15:
        rescale_moments = Structure.length_moments / max_length_moments
        return rescale_moments * moments
    else:
        return None


def _convert_to_moment_to_3d_vector(moments):
    if moments is not None and moments.ndim == 1:
        moments = moments.reshape((len(moments), 1))
        no_new_moments = (0, 0)
        add_zero_for_xy_axis = (2, 0)
        moments = np.pad(moments, (no_new_moments, add_zero_for_xy_axis))
    return moments


def _max_length_moments(moments):
    if moments is not None:
        return np.max(np.linalg.norm(moments, axis=1))
    else:
        return 0.0
