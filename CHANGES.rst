=========
Changelog
=========

2.0 (unreleased)
================

Upgrading from 1.x
------------------
- Python 3.11 or later is required.
- The ``pv_to_sip`` command now adds the reverse SIP coefficients (``AP_*``,
  ``BP_*``) by default, matching the ``pv_to_sip()`` function. In 1.x the
  command added them only with ``--add_reverse``. Use ``--no-add-reverse`` to
  get the old output.
- On numpy 1.x, the constant and linear reverse coefficients (``AP_0_0``,
  ``AP_1_0``, ``AP_0_1`` and the ``BP`` equivalents) change by up to about
  1e-5 pixels. They now match what numpy 2.x gives (see Numerics).
- The ``version`` attributes of the ``sip_tpv.pv_to_sip`` and
  ``sip_tpv.sip_to_pv`` modules have been removed. Use ``sip_tpv.__version__``.

The Python functions ``sip_to_pv()`` and ``pv_to_sip()`` take the same
arguments as before.

Command-line tools
------------------
- New ``sip_to_pv`` and ``pv_to_sip`` commands are installed with the package.
  ``python -m sip_tpv.sip_to_pv`` and ``python -m sip_tpv.pv_to_sip`` still
  work.
- Options are now spelled with hyphens (``--preserve-sip``, ``--write-tan-sip``,
  ...). The 1.x underscore spellings are still accepted.
- ``--extension`` accepts an HDU index or an EXTNAME. In 1.x only names worked.
- An extension that doesn't exist gives a usage error instead of a traceback.
- ``--write-tan`` and ``--write-tan-sip`` can no longer be combined.

Numerics
--------
- The reverse SIP fit passes ``rcond=None`` to ``numpy.linalg.lstsq``, so
  numpy 1.x and 2.x give the same coefficients (to about 1e-16).
- ``numpy.matrix`` is no longer used, which removes the
  ``PendingDeprecationWarning`` from numpy. Results are unchanged. (#9)

Packaging
---------
- Built from ``pyproject.toml`` with a ``src/`` layout; ``setup.py`` is gone.
- Minimum dependency versions: numpy 1.23.2, astropy 5.2, sympy 1.11 and
  click 8.0 (new dependency, used by the command-line tools).
- Tested on Python 3.11 to 3.15 on Linux, and on Python 3.15 on macOS and
  Windows.
- License metadata uses the SPDX identifier ``BSD-3-Clause``.

Documentation
-------------
- Restored the full docstrings for ``sip_to_pv()``, ``pv_to_sip()`` and the
  helpers in ``reverse.py``, and corrected the reverse-order defaults in the
  ``pv_to_sip()`` docstring (``A_ORDER``/``B_ORDER``, not 4).
- The license and citation text in each module is now its real docstring.
- The README describes the command-line tools.
- Fixed a typo in ``LICENSE.txt``; copyright years updated to 2012-2026.

Development
-----------
- New tests for the command-line tools, and a regression test that compares
  converted coefficients with output from the 1.2 release.
- GitHub Actions run the tests on every pull request, including Python 3.11
  with the minimum dependency versions.
- Releases are published to PyPI from GitHub Actions with Trusted Publishing
  when a version tag is pushed.

1.2 (2026-11-04)
================
- Replace ``pkg_resources`` with ``importlib.metadata`` to get the version.
  ``pkg_resources`` was removed in setuptools 82, which made
  ``import sip_tpv`` fail.
- Fix ``setup.py`` so the package builds again.
- Compare world coordinates in the tests with a tight tolerance instead of
  exact equality, so they pass with current numpy and astropy.

1.1 (2017-06-21)
================
- Last release before 1.2.
