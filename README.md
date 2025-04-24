## 📆 PAR_BGC_Argo: Calculation of unbiased PAR from multispectral irradiance profiles

Code routines to calculate PAR from multispectral irradiance profiles
Implements a method that delivers unbiased PAR estimates, based on two-layer neural networks, formulable in a small number of matrix equations, and thus exportable to any software platform. The method was calibrated with a dataset of hyperspectral acquired by new types of BioGeoChemical (BGC)-Argo floats deployed in a variety of open ocean locations, representative of a wide range of bio-optical properties. This procedure was repeated for several band configurations, including those existing on multispectral radiometers presently the standard for the BGC-Argo fleet. Validation results against independent data were highly satisfactory, displaying minimal uncertainties across a wide PAR range.
This release refines the previous one, described in the L&O Methods publication.
In the previous release, an increased error between the modeled and the reference PAR was acknowledged for the highest part of the range. No systematic traits were identified. However, when plotting such errors against depth, the pattern is quite systematic. Therefore, such remaining biased has been quantified and removed in this release. As a consequence, depth is now an input value.
In the previous release, the PAR range had been not properly delimited, and took all available wavelengths in the calibration dataset, which were ~ 330 nm to 780 nm. This might have caused a bias in the algorithm calibration. Such issue has been solved in this release, by delimiting PAR calculations to the range 400 nm to 700 nm.
Furthermore, in addition to all four band configurations of the previous release, a fifth one, for the bands 380, 443, 490 and 560 nm has been created.
---

## 🚀 Main Features

- **Fast**: The implementation has been efficiently computed as a set of matrix operations.
- **Multiband**: Code versions for several multispectral configurations are provided.
- **Uncertainty Estimation**: Provides uncertainty estimates for the retrieved PAR.
- **Residual bias compensation**: Depth was found as a good predictor of the estimate residual, and it was used for its removal.
- **Completeness**: Each function is self-contained.
- **MATLAB/Python**: MATLAB and Python codes are provided.

---

## 📚 Basic Usage

Please refer to the help provided inside each function

---

## 📄 Documentation

For a detailed description of the functions and parameters, see the LOM paper:  https://doi.org/10.1002/lom3.10673

---

---

## 📝 License

This project is licensed under the [GPL-3.0 License](https://www.gnu.org/licenses/gpl-3.0.html).

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request to suggest improvements or report bugs.

---

**Code versions**
| Version | Location | Key Differences |
|---------|----------|-----------------|
| MATLAB  | `/MATLAB` | Original algorithm |
| Python  | `/o25` | Open-source, NumPy/SciPy port |

---

## 📢 Contact

Inquiries to jaime.pitarch@cnr.it.