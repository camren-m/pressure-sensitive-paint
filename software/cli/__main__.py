import core.calibration as calibration
import click
from pathlib import Path

@click.group()
def cli():
    pass

@cli.command(
    help = "Generates a calibration curve (pressure v. intensity) based on a set of .cine files of known pressure values."
)
@click.option(
    "--data-dir",
    "data_dir_path",
    type = click.STRING,
    help = "Folder containing .cine files at varying pressures",
    required=True,
)
@click.option(
    "--atmospheric-pressure",
    "atmospheric_pressure",
    type = click.FloatRange(500, 2000),
    help = "The atmospheric pressure value in hPa",
    default = 1013.25,
)
def generate_calibration(data_dir_path, atmospheric_pressure):
    data_dir = Path(data_dir_path).resolve()
    if not data_dir.exists():
        raise Exception(f"Failed to find path {data_dir_path}")

    captures = calibration.ExperimentalCapture.from_folder(data_dir)
    cropped_captures = calibration.prompt_crop_captures(captures)

    pressure_intensity_map = calibration.create_pressure_intensity_map(cropped_captures)
    calibration.plot_pressure_v_intensity(pressure_intensity_map)

if __name__ == "__main__":
    cli()