### This repo provides a collection of Python sources to convert radiance/reflectance multispectral images from the spectral domain to a given color space

<p style='font-weight: bold'>It contains the following sources:</p>
<div class="snippet-clipboard-content notranslate position-relative overflow-auto" data-snippet-clipboard-copy-content="Codes
├── ..."><pre lang="none" class="notranslate"><code><strong>Spectral2Color</strong>
├── ColorTransform.py: implements the "Transform" class that provides different functions for multispectral to RGB/RGB-NIR space conversion
├── radiance2adobeRGB.py: transform multispectral radiance images to Adobe RGB color space
├── radiance2CIERGB.py: transform multispectral radiance images from spectral domain to CIE RGB color space
├── radiance2sRGB.py: transform multispectral radiance images to sRGB color space
├── reflectance2adobeRGB.py: transform multispectral reflectance images to Adobe RGB color space
├── reflectance2CIERGB.py: transform multispectral reflectance images to CIE RGB color space
├── reflectance2rgb_radiance_Sonyimx135.py: transform multispectral reflectance images to RGB radiance images based on SONY IMX135 color camera SSFs and a specified illuminant 
├── reflectance2rgbnir_imecssfs.py: transform multispectral reflectance images to RGB-NIR reflectance images using <a href="https://www.imechyperspectral.com/en/cameras/ximea-snapshot-rgb-nir#specs" target ="_top">IMEC</a> RGB-NIR camera SSFs
├── reflectance2rgbnir_monnossfs.py: transform multispectral reflectance images to RGB-NIR reflectance images using <a href="https://www.semanticscholar.org/paper/Single-Sensor-RGB-NIR-Imaging%3A-High-Quality-System-Monno-Teranaka/07d7936bcd3cde230167fcb1352cd3a8b918e6d6" target="_top">Monno et al.</a> RGB-NIR camera SSFs
</code></pre></div>


### Folder "utils"
<p>It contains the spectral power distribution (SPD) of different illumination sources, as well as several camera spectral sensitivity functions (SSFs).</p>
