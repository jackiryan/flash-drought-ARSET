# Flash Drought NASA ARSET Course
### Agricultural Flash Drought Detection Using Solar-Induced Fluorescence (SIF) and Soil Moisture Data

**ARSET Course Link**: https://www.earthdata.nasa.gov/learn/trainings/agricultural-flash-drought-detection-using-solar-induced-fluorescence-sif-soil

This repository contains code the NASA ARSET course on Agricultural Flash Drought. The training is comprised of two exercises with additional material in a third notebook:

1. **1_drought_indicators.ipynb**: Methods for retrieving SIF and Soil Moisture data from the OCO-2 and SMAP missions, and deriving time series of the Rapid Change Index (RCI) and Soil Water Deficit Index (SWDI) based on these measurements. These indicators will be used to assess flash droughts in two real-world scenarios.
2. **2_detection.ipynb**: We will apply the indicators we derived in the first exercise to build a framework for detecting flash droughts in general. The effectiveness of different definitions of flash drought will be compared.
3. **3_appendix.ipynb**: An appendix notebook is included to explain how the contents of the `notebooks/inputs/` directory were derived. These inputs are the result of steps that take a long time to run and so were pre-processed to improve the experience of running the main two notebooks.

## Youtube Links

* Part 1: https://www.youtube.com/watch?v=ZMQZBUs6n0g
* Part 2: https://www.youtube.com/watch?v=6nlZeWZgHXs

Note that some material shown in the video walkthroughs has changed since recording in response to issues encountered by participants and course feedback. If you would like to follow along on your own, you should use the current version of the Jupyter Notebooks rather than trying to copy the code from the Youtube recordings.

## Learning Objectives

By the end of this course, you will learn how to:

* Identify the causes, risks and consequences of flash droughts as compared to slowly-evolving droughts.
* Recognize how SIF data can be used for the detection and prediction of flash drought events.
* Compare the advantages and limitations of SIF, soil moisture and meteorological data to detect the onset of flash drought.
* Synthesize Solar-Induced Fluorescence Rapid Change Index (SIF-RCI) time series data for a selected region using a provided Jupyter Notebook.
* Retrieve and manipulate SMAP Soil Moisture (SM) and land process model-based Soil Water Deficit Index (SWDI) time series, and compare with SIF-RCI time series data using a provided Jupyter Notebook.

## Installation

1. **Install Python or Anaconda.** If you do not already have one of these installed:
   * **Python users:** Download Python 3.12 or later from [python.org](https://www.python.org/downloads/) and run the installer.
   * **Anaconda users:** Download Anaconda from [anaconda.com](https://www.anaconda.com/download) and run the installer.

2. **Download the course materials.** Click the green "Code" button at the top of this page, select "Download ZIP", and unpack the zip file wherever you like. The unpacked folder will be called `flash-drought-ARSET-main`. Alternatively, you can clone this repository if you know how to do so.

3. **Set up your NASA Earthdata account.** If you do not already have an account, go to [Earthdata Login](https://urs.earthdata.nasa.gov/) and create one. Remember the username and password you used.

4. **Open Terminal (MacOS) or PowerShell (Windows) and navigate to the folder where you unpacked the code** (`flash-drought-ARSET-main` if you downloaded the ZIP, or `flash-drought-ARSET` if you cloned the repository). For example, if you unpacked the ZIP in your Downloads folder, run:
   * MacOS/Linux: `cd ~/Downloads/flash-drought-ARSET-main`
   * Windows: `cd ~\Downloads\flash-drought-ARSET-main`

5. **Run the setup script to install dependencies and open the Jupyter Notebook in a browser window.**
* Anaconda users on Windows, run `.\setup-conda.ps1` in PowerShell.
* Python users on Windows, run `.\setup.ps1` in PowerShell.
* Anaconda users on MacOS/Linux, run `bash setup-conda.sh` in Terminal.
* Python users on MacOS/Linux, run `bash setup.sh` in Terminal.

> **Windows users:** If you see an error saying that running scripts is disabled, or that the script "is not digitally signed", run the setup script this way instead (Anaconda users: replace `setup.ps1` with `setup-conda.ps1`):
> ```powershell
> powershell -ExecutionPolicy Bypass -File .\setup.ps1
> ```

If you would like to open the Jupyter Notebook on subsequent uses once your environment is set up, simply run the setup script again.


## Common Questions

### Why not use Google CoLab?
Google CoLab is easy to set up and share but has limits on file storage and does not persist files between sessions unless set up to mount data on Google Drive. To avoid any unnecessary complexity with Google Drive storage, this notebook is intended to run locally on your computer.

### Why are there Python scripts in addition to the Jupyter Notebooks?
The code in the Jupyter notebooks in this course is designed to be focused around **conceptual understanding of the scientific technique** rather than implementation details of certain tasks. Helper functions, such as code for downloading data and rendering the interactive visualization, have been moved outside the notebook into standalone Python files.

### What is the Appendix notebook? Do I need to run it?
Some steps involved in processing the data, such as producing SIF climatology or virtualizing terabytes of SMAP data, take a long time to run (>10 minutes) and require a fast internet connection. The `3_appendix.ipynb` notebook explains these steps in detail, but it has already been run for you and its outputs have been saved in the `notebooks/inputs/` directory. **You do not need to run this notebook to complete the course**, but you may wish to run it on your own if you want to experiment with the flash drought technique outside of the case studies we discuss.

### How do I use these notebooks for my own analysis?
* All output files are saved to the `notebooks/data/` directory in GeoTIFF format for map-projected data, or CSV format for time series. If you prefer to use GIS software, the GeoTIFF files can readily be loaded into your application of choice.
* **If you wish to use the technique from this course in your own region and time period of choice**, you will need to follow both the steps in the Appendix and the main two notebooks, replacing the date ranges and bounding box (bbox) coordinates with your own values where marked in the code cells. [A guide may be included to help with this process.]
* **If you wish to do Near Real-Time (NRT) analysis with present data**, you will need to use a different SIF dataset other than GOSIF, e.g. [TROPOMI SIF](https://data-portal.s5p-pal.com/products/troposif.html), since GOSIF data are not updated on a timely cadence (only data through the end of 2024 are available as of the writing of this course). Likewise, the SMAP L4 virtual dataset that we use for deriving SWDI does not virtualize data past 2025, so you will need to use the Appendix notebook to virtualize more recent data. NASA's SMAP L4 collection has a 3-day latency.


## Contact

Please email [Jacqueline Ryan](mailto:Jacqueline.Ryan@jpl.nasa.gov) at JPL for any questions about the code in this course.

## Citations

[1] Lisonbee, J., Woloszyn, M., & Skumanich, M. (2021). Making sense of flash drought:
definitions, indicators, and where we go from here. Journal of Applied and Service Climatology, 1, https://doi.org/10.46275/JOASC.2021.02.001

[2] Christian, J. I., Hobbins, M., Hoell, A., et al. (2024). Flash drought: A state of the science review. WIREs Water, 11(3), e1714, https://doi.org/10.1002/wat2.1714

[3] Mohammadi, K., Jiang, Y. & Wang, G. (2022). Flash drought early warning based on the trajectory of solar-induced chlorophyll fluorescence, Proc. Natl. Acad. Sci. U.S.A. 119 (32) e2202767119, https://doi.org/10.1073/pnas.2202767119

[4] Mohammadi, K., & Wang, G. (2025). Impact Matters: Detection and Early Warning of Agriculturally Impactful Flash Droughts. Bull. Amer. Meteor. Soc., 106, E752–E769, https://doi.org/10.1175/BAMS-D-24-0143.1

[5] Yoshida, Y., Joiner, J., Tucker, C., et al. (2015). The 2010 Russian drought impact on satellite measurements of solar-induced chlorophyll fluorescence: Insights from modeling and comparisons with parameters derived from satellite reflectances. Remote Sensing of Environment, 166, 163-177. https://doi.org/10.1016/j.rse.2015.06.008

[6] He, M., Kimball, J. S., Yi, Y., et al. (2019). Impacts of the 2017 flash drought in the US Northern plains informed by satellite-based evapotranspiration and solar-induced fluorescence. Environ. Res. Lett., 14, https://doi.org/10.1088/1748-9326/ab22c3

[7] Kimball, J. S., Jones, L., Jensco, K., He, M., Maneta, M. P., & Reichle, R. H. (2019). SMAP L4 Assessment of the US Northern Plains 2017 Flash Drought. International Geoscience and Remote Sensing Symposium (IGARSS), 5366-5369.

[8] Sehgal, V., Gaur, N., & Mohanty, B. P. (2021). Global flash drought monitoring using surface soil moisture. Water Resources Research, 57, e2021WR029901, https://doi.org/10.1029/2021WR029901

[9] Brust, C., Kimball, J. S., Maneta, M. P., Jencso, K., He, M., & Reichle, R. H. (2021). Using SMAP Level-4 soil moisture to constrain MOD16 evapotranspiration over the contiguous USA. Remote Sensing of Environment, 255, 112277, https://doi.org/10.1016/j.rse.2020.112277

[10] Tang, S., Wang, S., Jiang, J., & Zheng, Y. (2026). Improved flash drought forecasting and attribution: A spatial-temporal causality-aware deep learning approach. Journal of Hydrology, 667, 134945, https://doi.org/10.1016/j.jhydrol.2026.134945

[11] Li, X. & Xiao, J. (2019). A global, 0.05-degree product of solar-induced chlorophyll fluorescence derived from OCO-2, MODIS, and reanalysis data. Remote Sensing, 11, 517. https://doi.org/10.3390/rs11050517

[12] Reichle, R., De Lannoy, G., Koster, R. D., Crow, W. T., Kimball, J. S., Liu, Q. & Bechtold, M. (2025). SMAP L4 Global 3-hourly 9 km EASE-Grid Surface and Root Zone Soil Moisture Geophysical Data. (SPL4SMGP, Version 8). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/T5RUATAQREF8

[13] Reichle, R., De Lannoy, G., Koster, R. D., Crow, W. T., Kimball, J. S., Liu, Q. & Bechtold, M. (2025). SMAP L4 Global 3-hourly 9 km EASE-Grid Surface and Root Zone Soil Moisture Land Model Constants. (SPL4SMLM, Version 8). [Data Set]. Boulder, Colorado USA. NASA National Snow and Ice Data Center Distributed Active Archive Center. https://doi.org/10.5067/PXQIBL2ALDZD

[14] Martínez-Fernández, J., González-Zamora, A., Sánchez, N., & Gumuzzio, A. (2015). A soil water based index as a suitable agricultural drought indicator. Journal of Hydrology, 522, 265-273. https://doi.org/10.1016/j.jhydrol.2014.12.051

[15] Malherbe, J., Moeletsi, M. E., Roffe, S. J., Beukes, P. J., & Masupha, T. E. (2025). The 2023/24 summer drought tracked by the Agricultural Drought Early Warning System (ADEWS) for South Africa. Environmental Development, 56, 101253. https://doi.org/10.1016/j.envdev.2025.101253

[16] Parazoo, N., Osman, M., Pascolini-Campbell, M., & Byrne, B. (2024). Antecedent conditions mitigate carbon loss during flash drought events. Geophysical Research Letters, 51, e2024GL108310. https://doi.org/10.1029/2024GL108310

[17] Osman, M., Zaitchik, B. F., Badr, H. S., Christian, J. I., Tadesse, T., Otkin, J. A., & Anderson, M. C. (2021). Flash drought onset over the contiguous United States: Sensitivity of inventories and trends to quantitative definitions. Hydrology and Earth System Sciences, 25(2), 565–581. https://doi.org/10.5194/hess-25-565-2021



## Acknowledgements

Special thanks to Xiaolan Xu from the SMAP team at JPL for her technical guidance and Rolf Reichle at NASA Goddard for his work in developing the SMAP Level 4 soil moisture dataset used in this course.

**GOSIF Dataset:** Used with permission from the authors Xing Li and Jingfeng Xiao from the University of New Hampshire.

**All Code:** Copyright 2026, by the California Institute of Technology. ALL RIGHTS RESERVED. United States Government Sponsorship acknowledged. Any commercial use must be negotiated with the Office of Technology Transfer at the California Institute of Technology.
