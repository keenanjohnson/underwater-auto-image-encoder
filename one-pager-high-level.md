# AI/ML_image_processing

## Background
We equip our ROV with two downward-facing GoPro HERO12 cameras to take 27.3MP survey images of the seafloor ([see custom settings here](https://www.dropbox.com/scl/fi/fiv13iwlppctow3ocjppy/settings_GoProLabs.JPG?rlkey=76fbpjrr68iau1sbsm6czf334&st=4fuixxwq&dl=0)), with the aim to photograph algae, invertebrates, fishes, and the underlying substrate.
The cameras capture RAW images (GoPro's General Purpose RAW, i.e., .GPR file; info [linked here](https://github.com/gopro/gpr)) at 3-second intervals, leaving us with hundreds if not thousands of unedited GPR images (the GoPro also captures a JPEG preview of each GPR photo, though these are not edited, as the GPR enables a much greater degree of post-processing).
While we batch edit these images in Adobe Lightroom Classic, each image often requires individual tuning before any meaningful data can be extracted from it with subsequent programs.
This individual editing is our workflow's most rate-limiting step. 

Additionally, you can access 11 example "input" photos (both GPR and the *JPEG preview) and the corresponding "output" (JPEG) [here](https://github.com/Seattle-Aquarium/CCR_development/tree/rmt_edits/files/ML_image_processing) on this repo (*the JPEG preview is only included to give you a visual idea of the GPR, as GPR photos need to be opened in software like Adobe Lightroom Classic).

## Existing workflow in Adobe Lightroom Classic
We start with a folder containing all GPR photos from a single ROV survey, or "transect." 
To process these photos, we 
(**1**) apply the denoise feature set to 55 to all of the images; we 
(**2**) choose a single image, crop it to dimensions 4606 x 4030, select a white patch to adjust white balance, and make any other necessary adjustments to Tone settings (exposure, highlights, shadows, whites, blacks). 
We (**3**) copy the settings from this single image and paste to all others within the transect to achieve a baseline for the entire image set.
We (**4**) then need to review each photo individually, tweaking Tone and, when necessary, Presence settings (texture, clarity, dehaze) to account for variation across images--and herein lies the most time-consuming step.
The final results are JPEG images that have been denoised, cropped, color-corrected, and brightened significantly.
For example, you can see a fully processed transect [here](https://www.dropbox.com/scl/fo/nkgka51g6zmk94c3je1zm/APA28IzNJSZ-_4uRkBHgLk0?rlkey=p7knm31b0la2kudx235fx3h72&st=ummi5snl&dl=0). 

## The Problem
The time it takes to edit a single transect's worth of imagery is the most time-consuming step in our entire workflow.
With our current methods, when we clean up one image and apply those settings to the whole transect--step (**3**) above--only images similar to the baseline image are processed to a satisfactory degree. 
Given that our images are recorded as the ROV moves across the seafloor, they capture variation in the physical environment. 
Therefore, substrate type, algae cover, depth, etc. are all factors that can affect the brightness, color, clarity, and other attributes of an image.
To ensure the quality of our imagery remains consistent, we cycle through and edit each image individually to account for this variation. 

## The Proposed Solution
Given that we have thousands of input and output files (unprocessed GPRs, and processed JPEGs), we seek to incorporate some form of machine learning to cut down on editing time.
It is our hope that our manual edits across many images will have captured a sufficient amount of variation such that an AI/ML framework could reproduce our individualized edits on unseen imagery. 
There appear to be some AI-powered image editing software available, such as [Aftershoot](https://aftershoot.com/edit/) or [Imagen](https://imagen-ai.com/).
Additionally, a framework in Python could likely be established. 
There are a variety of options available, and we are unsure of which options ought to be explored in more detail or are best suited for underwater imagery. 

And please see https://github.com/Seattle-Aquarium/CCR_development/issues/29 for active discussion of various approaches / considerations re: this problem. 