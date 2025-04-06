## 🗂️ Datasets to evaluate zero-shot performance
We curated 32 publicly available medical imaging datasets from diverse platforms including [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net), [Kaggle](https://www.kaggle.com), [Zenodo](https://zenodo.org), [IEEE DataPort](https://ieee-dataport.org), and [Grand Challenge](https://grand-challenge.org). To ensure fair zero-shot evaluation, datasets used in the training of the selected six foundation models were excluded.

Our benchmark consists of:
- 49,401 2D images and 1,624 3D volumes (≈54,444 slices)
- Covering 10 imaging modalities, including CT, MR, X-ray, dermoscopy, endoscopy, and histopathology
- Spanning 4 major anatomical regions and 16 distinct anatomies, including key lesions like skin cancer and breast tumors


This diverse and carefully selected dataset suite enables a robust and realistic evaluation of model generalizability in medical image segmentation.

<table border="2" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Dataset</th>
      <th>Dimension</th>
      <th>Modality</th>
      <th>Segmentation Targets</th>
      <th># of images/scans</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="https://uwaterloo.ca/vision-image-processing-lab/research-demos/skin-cancer-detection">UWater Skin Cancer</a></td>
      <td>2D</td>
      <td>Dermoscopy</td>
      <td>Skin cancer</td>
      <td>180</td>
    </tr>
    <tr>
      <td><a href="https://www.kaggle.com/datasets/nguyenvoquocduong/etis-laribpolypdb">ETIS-LaribPolypDB</a></td>
      <td>2D</td>
      <td>Endoscopy</td>
      <td>Polyp</td>
      <td>196</td>
    </tr>
    <tr>
      <td><a href="https://autolaparo.github.io/">AutoLaparo</a></td>
      <td>2D</td>
      <td>Endoscopy</td>
      <td>Uterus, Surgical instruments</td>
      <td>1,800</td>
    </tr>
    <tr>
      <td><a href="https://drac22.grand-challenge.org/Description/">DRAC2022</a></td>
      <td>2D</td>
      <td>OCT</td>
      <td>Diabetic Retinopathy Lesions</td>
      <td>106</td>
    </tr>
    <tr>
      <td><a href="https://segpc-2021.grand-challenge.org/Dataset/">SegPC2021</a></td>
      <td>2D</td>
      <td>Microscopy</td>
      <td>Cytoplasm, Nucleus</td>
      <td>497</td>
    </tr>
    <tr>
      <td><a href="https://mitoem.grand-challenge.org/">MitoEM</a></td>
      <td>2D</td>
      <td>Microscopy</td>
      <td>Mitochondria Instance</td>
      <td>230</td>
    </tr>
    <tr>
      <td><a href="https://figshare.com/articles/dataset/PAPILA/14798004/1">PAPILA</a></td>
      <td>2D</td>
      <td>Fundus</td>
      <td>Optic disc and cup</td>
      <td>488</td>
    </tr>
    <tr>
      <td><a href="https://ravir.grand-challenge.org/">RAVIR</a></td>
      <td>2D</td>
      <td>Fundus</td>
      <td>Vein, artery</td>
      <td>23</td>
    </tr>
    <tr>
      <td><a href="https://ieee-dataport.org/documents/pathological-images-gland-segmentation">PathologyImagesForGlandSeg</a></td>
      <td>2D</td>
      <td>Histopathology</td>
      <td>Gland</td>
      <td>20,000</td>
    </tr>
    <tr>
      <td><a href="https://github.com/twpkevin06222/Gland-Segmentation">GlaS@MICCAI2015</a></td>
      <td>2D</td>
      <td>Histopathology</td>
      <td>Adenocarcinomas</td>
      <td>165</td>
    </tr>
    <tr>
      <td><a href="https://monusac-2020.grand-challenge.org/Home/">MoNuSAC2020</a></td>
      <td>2D</td>
      <td>Histopathology</td>
      <td>Cell</td>
      <td>100</td>
    </tr>
    <tr>
      <td><a href="https://wsss4luad.grand-challenge.org/">WSSS4LUAD</a></td>
      <td>2D</td>
      <td>Histopathology</td>
      <td>Tissue</td>
      <td>120</td>
    </tr>
    <tr>
      <td><a href="https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database">COVID-19 Radiography</a></td>
      <td>2D</td>
      <td>X-ray</td>
      <td>Lung, COVID-19 infection, lung opacity, viral pneumonia</td>
      <td>21,165</td>
    </tr>
    <tr>
      <td><a href="https://arcade.grand-challenge.org/">ARCADE</a></td>
      <td>2D</td>
      <td>X-ray</td>
      <td>Coronary Artery Disease</td>
      <td>2997</td>
    </tr>
    <tr>
      <td><a href="https://ieee-dataport.org/documents/re-curated-breast-imaging-subset-ddsm-dataset-rbis-ddsm">RBIS-DDSM</a></td>
      <td>2D</td>
      <td>X-ray</td>
      <td>Breast Cancer</td>
      <td>689</td>
    </tr>
    <tr>
      <td><a href="https://data.mendeley.com/datasets/zm6bxzhmfz/1">Xray_hip</a></td>
      <td>2D</td>
      <td>X-ray</td>
      <td>Femur, Ilium</td>
      <td>140</td>
    </tr>
    <tr>
      <td><a href="https://qamebi.com/breast-ultrasound-images-database/">QAMEBI</a></td>
      <td>2D</td>
      <td>Ultrasound</td>
      <td>Benign &amp; Malignant breast lesion</td>
      <td>232</td>
    </tr>
    <tr>
      <td><a href="https://data.mendeley.com/datasets/vckdnhtw26/1">BUSC</a></td>
      <td>2D</td>
      <td>Ultrasound</td>
      <td>Benign &amp; Malignant breast lesion</td>
      <td>250</td>
    </tr>
    <tr>
      <td><a href="https://data.mendeley.com/datasets/4gcpm9dsc3/1">USFetalHead</a></td>
      <td>2D</td>
      <td>Ultrasound</td>
      <td>Stomach, Liver, Vein</td>
      <td>1,588</td>
    </tr>
    <tr>
      <td><a href="https://tdsc-abus2023.grand-challenge.org/">TDSC-ABUS2023</a></td>
      <td>3D</td>
      <td>Ultrasound</td>
      <td>Breast Tumor</td>
      <td>100</td>
    </tr>
    <tr>
      <td><a href="https://curious2022.grand-challenge.org/data/">RESECT-SEG</a></td>
      <td>3D</td>
      <td>Ultrasound</td>
      <td>Brain Tumor, Resection Cavity</td>
      <td>69</td>
    </tr>
    <tr>
      <td><a href="https://atlas-challenge.u-bourgogne.fr/">ATLAS2023</a></td>
      <td>3D</td>
      <td>MR</td>
      <td>Liver, Tumor</td>
      <td>60</td>
    </tr>
    <tr>
      <td><a href="https://crossmoda2022.grand-challenge.org/dataset">CrossMoDA2022</a></td>
      <td>3D</td>
      <td>MR</td>
      <td>Brain tumor, Cochela</td>
      <td>210</td>
    </tr>
    <tr>
      <td><a href="https://ieee-dataport.org/documents/pituitary-adenoma-mri-segmentation-dataset">PASeg</a></td>
      <td>3D</td>
      <td>MR</td>
      <td>Pituitary Adenoma</td>
      <td>55</td>
    </tr>
    <tr>
      <td><a href="https://shifts.grand-challenge.org/">Shifts2022</a></td>
      <td>3D</td>
      <td>MR</td>
      <td>White Matter Multiple Sclerosis (MS) lesion</td>
      <td>371</td>
    </tr>
    <tr>
      <td><a href="https://asoca.grand-challenge.org/">ASOCA</a></td>
      <td>3D</td>
      <td>CT</td>
      <td>Coronary Artery</td>
      <td>40</td>
    </tr>
    <tr>
      <td><a href="https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=9325794593257945d20474acc83148a18de7f16d11c52341">Adrenal-ACC-Ki67-Seg</a></td>
      <td>3D</td>
      <td>CT</td>
      <td>Adrenal mass</td>
      <td>53</td>
    </tr>
    <tr>
      <td><a href="https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230229">HCC-TACE-Seg</a></td>
      <td>3D</td>
      <td>CT</td>
      <td>Liver</td>
      <td>104</td>
    </tr>
    <tr>
      <td><a href="TBAD	https://www.kaggle.com/datasets/xiaoweixumedicalai/imagetbad">ImageTBAD</a></td>
      <td>3D</td>
      <td>CT</td>
      <td>Aorta</td>
      <td>100</td>
    </tr>
    <tr>
      <td><a href="https://ieee-dataport.org/documents/ct-training-and-validation-series-3d-automated-segmentation-inner-ear-using-u-net">InnerEarSeg</a></td>
      <td>3D</td>
      <td>CT</td>
      <td>Inner Ear</td>
      <td>341</td>
    </tr>
    <tr>
      <td><a href="https://segrap2023.grand-challenge.org/">SegRap2023_Task2</a></td>
      <td>3D</td>
      <td>CT</td>
      <td>GTVp, GTVnd</td>
      <td>100</td>
    </tr>
  </tbody>
</table>

