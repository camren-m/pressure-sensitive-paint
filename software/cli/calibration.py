import click
from pathlib import Path
from cli.group import cli
from core import calibration

@cli.command(
    help = "Generates a calibration curve (pressure v. intensity) based on a set of .cine files of known pressure values."
)
@click.option(
    "--data-dir",
    "data_dir_path",
    type = click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path, readable=True, resolve_path=True),
    help = "Folder containing .cine files at varying pressures.",
    required=True,
)
@click.option(
    "--output",
    "output_file_path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path, writable=True, resolve_path=True),
    help = "File to output the calibration coefficients to. Will be overwritten.",
    default = "calibration.poly"
)
@click.option(
    "--atmospheric-pressure",
    "atmospheric_pressure",
    type = click.FLOAT,
    help = "The atmospheric pressure value in Pa",
    default = 101325.0,
)
def generate_calibration(data_dir_path: Path, output_file_path: Path, atmospheric_pressure: float):
    captures = calibration.ExperimentalCapture.from_folder(data_dir_path)
    wind_off_capture = next((f for f in captures if f.rel_pressure == 0.0), None)
    assert wind_off_capture is not None, "No wind-off capture found in data directory. Please include a .cine file with a pressure of 0.00 Pa"
    aligned_captures = [f.align(wind_off_capture) for f in captures]
    cropped_captures = calibration.prompt_crop_captures(aligned_captures)

    pressure_intensity_map = calibration.create_pressure_intensity_map(cropped_captures, atmospheric_pressure)
    calibration_curve = calibration.plot_pressure_v_intensity(pressure_intensity_map, atmospheric_pressure).convert()

    with open(output_file_path, "w") as output_file:
        output_file.write(f"{calibration_curve.coef[0]},{calibration_curve.coef[1]},{atmospheric_pressure}")