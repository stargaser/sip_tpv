from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits
from click.testing import CliRunner

from sip_tpv.pv_to_sip import cli as pv_to_sip_cli
from sip_tpv.sip_to_pv import cli as sip_to_pv_cli

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def sip_file(tmp_path):
    """FITS file with the SIP header in the primary HDU and in an 'SCI' extension."""
    header = fits.Header.fromtextfile(DATA_DIR / "IRAC_3.6um_sip.txt")
    path = tmp_path / "sip.fits"
    fits.HDUList([fits.PrimaryHDU(header=header),
                  fits.ImageHDU(header=header, name="SCI")]).writeto(path)
    return path


@pytest.fixture
def tpv_file(tmp_path):
    """FITS file with the TPV header in the primary HDU.

    The reverse SIP fit needs NAXIS1/NAXIS2, so include a (uint8) image.
    """
    header = fits.Header.fromtextfile(DATA_DIR / "PTF_r_chip01_tpv.txt")
    data = np.zeros((header["NAXIS2"], header["NAXIS1"]), dtype=np.uint8)
    path = tmp_path / "tpv.fits"
    fits.PrimaryHDU(data=data, header=header).writeto(path)
    return path


def run(cli, *args):
    return CliRunner().invoke(cli, [str(a) for a in args])


def test_sip_to_pv_cli(sip_file, tmp_path):
    out = tmp_path / "out.fits"
    result = run(sip_to_pv_cli, sip_file, out)
    assert result.exit_code == 0, result.output
    header = fits.getheader(out, 0)
    assert header["CTYPE1"] == "RA---TPV"
    assert "PV1_1" in header
    assert "A_ORDER" not in header


def test_sip_to_pv_cli_options(sip_file, tmp_path):
    out = tmp_path / "out.fits"
    result = run(sip_to_pv_cli, sip_file, out, "--write-tan-sip", "--preserve-sip")
    assert result.exit_code == 0, result.output
    header = fits.getheader(out, 0)
    assert header["CTYPE1"] == "RA---TAN-SIP"
    assert "PV1_1" in header
    assert "A_ORDER" in header


@pytest.mark.parametrize("extension", ["1", "SCI"])
def test_extension_by_index_or_name(sip_file, tmp_path, extension):
    out = tmp_path / "out.fits"
    result = run(sip_to_pv_cli, sip_file, out, "--extension", extension)
    assert result.exit_code == 0, result.output
    assert "PV1_1" in fits.getheader(out, "SCI")
    assert "PV1_1" not in fits.getheader(out, 0)


@pytest.mark.parametrize("extension", ["5", "NOPE"])
def test_missing_extension(sip_file, tmp_path, extension):
    result = run(sip_to_pv_cli, sip_file, tmp_path / "out.fits",
                 "--extension", extension)
    assert result.exit_code == 2
    assert "no extension" in result.output


def test_write_tan_options_exclusive(sip_file, tmp_path):
    result = run(sip_to_pv_cli, sip_file, tmp_path / "out.fits",
                 "--write-tan", "--write-tan-sip")
    assert result.exit_code == 2
    assert "mutually exclusive" in result.output


def test_overwrite(sip_file, tmp_path):
    out = tmp_path / "out.fits"
    assert run(sip_to_pv_cli, sip_file, out).exit_code == 0
    assert run(sip_to_pv_cli, sip_file, out).exit_code != 0
    assert run(sip_to_pv_cli, sip_file, out, "--overwrite").exit_code == 0


def test_pv_to_sip_cli(tpv_file, tmp_path):
    out = tmp_path / "out.fits"
    result = run(pv_to_sip_cli, tpv_file, out, "--add-reverse", "--aporder", 3)
    assert result.exit_code == 0, result.output
    header = fits.getheader(out, 0)
    assert header["CTYPE1"] == "RA---TAN-SIP"
    assert "A_ORDER" in header
    assert header["AP_ORDER"] == 3
    assert "PV1_1" not in header


@pytest.mark.parametrize("args, expected", [
    ([], True),
    (["--add-reverse"], True),
    (["--no-add-reverse"], False),
    (["--no_add_reverse"], False),
])
def test_add_reverse_default_on(tpv_file, tmp_path, args, expected):
    out = tmp_path / "out.fits"
    result = run(pv_to_sip_cli, tpv_file, out, *args)
    assert result.exit_code == 0, result.output
    assert ("AP_ORDER" in fits.getheader(out, 0)) == expected


def test_underscore_option_aliases(tpv_file, tmp_path):
    """The 1.x underscore spellings still work."""
    out = tmp_path / "out.fits"
    result = run(pv_to_sip_cli, tpv_file, out, "--preserve_tpv", "--add_reverse")
    assert result.exit_code == 0, result.output
    header = fits.getheader(out, 0)
    assert "PV1_1" in header
    assert "AP_ORDER" in header
