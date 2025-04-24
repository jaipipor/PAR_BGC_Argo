## 📆 PAR_BGC_Argo: Calculation of unbiased PAR from multispectral irradiance profiles

Code routines to calculate PAR from multispectral irradiance profiles
Implements a method that delivers unbiased PAR estimates, based on two-layer neural networks, formulable in a small number of matrix equations, and thus exportable to any software platform. The method was calibrated with a dataset of hyperspectral acquired by new types of BioGeoChemical (BGC)-Argo floats deployed in a variety of open ocean locations, representative of a wide range of bio-optical properties. This procedure was repeated for several band configurations, including those existing on multispectral radiometers presently the standard for the BGC-Argo fleet. Validation results against independent data were highly satisfactory, displaying minimal uncertainties across a wide PAR range.

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