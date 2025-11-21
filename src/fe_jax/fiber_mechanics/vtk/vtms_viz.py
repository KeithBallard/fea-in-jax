from ..vtms_structs import *

import vtk
from vtkmodules.numpy_interface.dataset_adapter import numpyTovtkDataArray as np2da


def to_vtk_polydata(vtms_obj: VTMSFabric | VTMSBundle) -> vtk.vtkPolyData:
    """
    Given an entire fabric, creates a VTK object to render each fiber as a poly line.
    """

    append_polydata = vtk.vtkAppendPolyData()

    fiber_id = 0
    for bundle_i in range(vtms_obj.get_n_bundles()):
        for fiber_i in range(vtms_obj.get_n_fibers_in_bundle(bundle_i)):
            points = vtk.vtkPoints()
            points.SetData(np2da(vtms_obj.get_fiber_points(bundle_i, fiber_i)))

            line_source = vtk.vtkPolyLineSource()
            line_source.SetPoints(points)
            line_source.Update()
            line_polydata: vtk.vtkPolyData = line_source.GetOutput()

            tow_id_array = vtk.vtkIntArray()
            tow_id_array.SetName("tow_id")
            tow_id_array.SetNumberOfComponents(1)
            tow_id_array.SetNumberOfTuples(line_polydata.GetNumberOfPoints())
            tow_id_array.Fill(bundle_i)

            fiber_id_array = vtk.vtkIntArray()
            fiber_id_array.SetName("fiber_id")
            fiber_id_array.SetNumberOfComponents(1)
            fiber_id_array.SetNumberOfTuples(line_polydata.GetNumberOfPoints())
            fiber_id_array.Fill(fiber_id)
            fiber_id += 1

            material_id_array = vtk.vtkIntArray()
            material_id_array.SetName("material_id")
            material_id_array.SetNumberOfComponents(1)
            material_id_array.SetNumberOfTuples(line_polydata.GetNumberOfPoints())
            material_id_array.Fill(vtms_obj.get_material_id(bundle_i))

            radii_array = vtk.vtkDoubleArray()
            radii_array.SetName("radius")
            radii_array.SetNumberOfComponents(1)
            radii_array.SetNumberOfTuples(line_polydata.GetNumberOfPoints())
            radii_array.Fill(vtms_obj.get_diameter(bundle_i) / 2.0)
            
            line_polydata.GetPointData().AddArray(tow_id_array)
            line_polydata.GetPointData().AddArray(fiber_id_array)
            line_polydata.GetPointData().AddArray(material_id_array)
            line_polydata.GetPointData().AddArray(radii_array)
            line_polydata.GetPointData().SetActiveScalars("radius")

            append_polydata.AddInputData(line_polydata)

    append_polydata.Update()
    tow_polydata: vtk.vtkPolyData = append_polydata.GetOutput()
    tow_polydata.GetPointData().SetActiveScalars("radius")
    return tow_polydata


def to_vtk_tubes(vtms_obj: VTMSFabric | VTMSBundle) -> vtk.vtkTubeFilter:
    """
    Given an entire fabric, creates a VTK object to render each fiber as a tube.
    """
    tube_filter = vtk.vtkTubeFilter()
    tube_filter.SetInputData(to_vtk_polydata(vtms_obj))
    tube_filter.SetNumberOfSides(10)
    tube_filter.SetCapping(True)
    tube_filter.SetVaryRadiusToVaryRadiusByAbsoluteScalar()
    tube_filter.Update()

    return tube_filter


def write_vtk(vtms_obj: VTMSFabric | VTMSBundle, filepath: str | Path, fibers_as_tubes = True):
    """
    TODO document

    NOTE Requires `vtk` Python library
    """
    filepath = Path(filepath)

    writer_vtk = vtk.vtkPolyDataWriter()
    if not fibers_as_tubes:
        vtk_output = to_vtk_polydata(vtms_obj)
        writer_vtk.SetInputData(vtk_output)
    else:
        vtk_output = to_vtk_tubes(vtms_obj)
        writer_vtk.SetInputConnection(vtk_output.GetOutputPort())
    writer_vtk.SetFileName(str(filepath))
    writer_vtk.Write()


def render(vtms_obj: VTMSFabric | VTMSBundle, fibers_as_tubes = True):
    """
    TODO document

    NOTE Requires `vtk` Python library
    """
    mesh_mapper = vtk.vtkPolyDataMapper()
    if not fibers_as_tubes:
        vtk_output = to_vtk_polydata(vtms_obj)
        mesh_mapper.SetInputData(vtk_output)
    else:
        vtk_output = to_vtk_tubes(vtms_obj)
        mesh_mapper.SetInputConnection(vtk_output.GetOutputPort())
    mesh_mapper.ScalarVisibilityOff()

    actor = vtk.vtkActor()
    actor.SetMapper(mesh_mapper)

    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)  # Set background color

    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)

    render_window.Render()
    interactor.Start()
