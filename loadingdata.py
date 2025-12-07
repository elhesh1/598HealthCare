import glob
import os
import pandas as pd
# loading basic data
# combines the path to the ISIC pics with the annotations by Bissoto et al. in a df

csv_path = r"C:\Users\sherm\OneDrive\Desktop\HealthCare\ShermanHealthProject\debiasing-skin-master\debiasing-skin-master\artefacts-annotation\isic_bias.csv"
image_dir = r"C:\Users\sherm\OneDrive\Desktop\HealthCare\ShermanHealthProject\ISIC2018_Task1-2_Training_Input\ISIC2018_Task1-2_Training_Input"

df1 = pd.read_csv(csv_path, sep=';')

df1['image_id'] = df1['image'].str.replace('.png', '', regex=False)
df1['image_id'] = df1['image_id'].str.strip()

image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))

df_imgs = pd.DataFrame({'full_path': image_paths})

temp = df_imgs['full_path'].str.split('\\').str[-1]
temp = temp.str.split('/').str[-1]
df_imgs['image_id'] = temp.str.replace(r"\..*$", "", regex=True)

df = df_imgs.merge(df1, on='image_id', how='left')

df.to_csv("isic_df.csv", index=False)
