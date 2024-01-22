import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool
from PIL import Image, ImageDraw
import cv2
# Specify the folder where the CSV files are located
folder_path = 'coldbrewLAST'
sample = "sample_submission.csv"
csv_save_dir = "csvs"
csv_name = 'cLAST05'
img_save_dir = 'cLAST05'

thread_num = 30

# List all CSV files in the folder
file_names = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
print("csv name : ", csv_name)
print("file_names : ",file_names)
# Threshold for the minimum number of votes required to select a class
threshold = 1  # Adjust this threshold as needed
print("threshold : ", threshold)


num_samples_per_group = 12
class_intensity = {
    0: 255,
    1: 40,
    2: 60,
    3: 80,
    4: 100,
    5: 120,
    6: 140,
    7: 160,
    8: 100,
    9: 200,
    10: 120,
    11: 20,
    255: 230
}
class_weights = {
    0: 2,  # Road
    1: 1,  # sidewalk
    2: 1,  # construction
    3: 2,  # fence
    4: 2,  # pole
    5: 2,  # traffic light
    6: 2,  # traffic sign
    7: 1,  # nature
    8: 2,  # sky
    9: 2,  # person
    10: 2, # rider
    11: 1  # car
}
print("weights : ", class_weights)
def rle_encode(mask):
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)
def apply_ellipse_filter(input_image):
    width, height = input_image.size
    background_color = (255, 255, 255)  # 배경색을 흰색으로 설정

    new_image = Image.new("RGBA", (width, height), background_color)
    draw = ImageDraw.Draw(new_image)

    ellipse_width = 1850//2
    ellipse_height = 1200//2
    ellipse_left = (width - ellipse_width) // 2
    ellipse_top = (height - ellipse_height) // 2
    ellipse_right = ellipse_left + ellipse_width
    ellipse_bottom = ellipse_top + ellipse_height - 100

    ellipse_color = (0, 0, 0, 0)  # 흰색 (RGB) 및 완전 불투명 (알파 채널)
    draw.ellipse((ellipse_left, ellipse_top, ellipse_right, ellipse_bottom), fill=ellipse_color)

    result_image = Image.alpha_composite(new_image, input_image.convert("RGBA"))

    result_cv2 = np.array(result_image)

    mask = np.zeros_like(result_cv2)  # 같은 크기의 빈 이미지 생성
    cv2.ellipse(mask, ((ellipse_left + ellipse_right) // 2, (ellipse_top + ellipse_bottom) // 2),
                (ellipse_width // 2, ellipse_height // 2), 0, 0, 360, (255, 255, 255), -1)  # 타원 내부 채우기
    result_cv2[mask == 0] = 255  # 타원 외부 픽셀 값을 흰색(255)으로 설정

    result_cv2 = cv2.cvtColor(result_cv2, cv2.COLOR_RGBA2GRAY)

    result_image = Image.fromarray(result_cv2)

    return result_image
    

# Function to process a single group
def process_group(group_index):
    # Initialize an empty array to store class counts for each pixel
    class_counts = np.zeros((540, 960, len(class_intensity)), dtype=np.uint8)
    result = []  # List to store mask_rle results
    
    for file_name in file_names:
        # Create the full path to the CSV file
        file_path = os.path.join(folder_path, file_name)

        df = pd.read_csv(file_path)
        group_df = df[group_index * num_samples_per_group: (group_index + 1) * num_samples_per_group]

        img_id = group_df.iloc[0]['id']
        # Create an image for combining masks with class-specific intensity levels
        mask_image = np.zeros((540, 960), dtype=np.uint8)
        

        
        for index, row in group_df.iterrows():
            mask_rle = row['mask_rle']

            if mask_rle != -1:
                mask = rle_decode(mask_rle, (540, 960))
        
                # Increment the count for the corresponding class
                class_index = index % 12
                class_counts[:, :, class_index] += mask * class_weights[class_index]

    # Determine the final class for each pixel based on the threshold
    final_mask_value = np.argmax(class_counts, axis=2)
    final_mask_value[class_counts.max(axis=2) < threshold] = 255  # Assign -1 to pixels below the threshold
    
    mask = np.array(final_mask_value)
    #필터씌우기
    mask = mask.astype(np.uint8)
    mask_img = Image.fromarray(mask)
    mask_img = apply_ellipse_filter(mask_img)
    mask = np.array(mask_img)
    for class_id in range(12):
        class_mask = (mask == class_id).astype(np.uint8)
        if np.sum(class_mask) > 0:  # If mask exists, encode
            mask_rle = rle_encode(class_mask)            
            result.append(mask_rle)

        else:  # If mask doesn't exist, append -1
            result.append(-1)
    
    # Save the image
    output_image_path = os.path.join(img_save_dir, f'mask_image_{group_index}.png')
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    plt.imsave(output_image_path, mask, cmap='viridis', vmin=0, vmax=len(class_intensity) - 1)
    
    # Return the result along with the group_index
    return group_index, result

if __name__ == '__main__':
    sample_df = pd.read_csv(sample)
    num_samples = len(sample_df)
    num_samples_per_group = 12
    num_groups = num_samples // num_samples_per_group

    results = []  # List to store the results for all groups

    with Pool(processes=thread_num) as pool:  # You can adjust the number of processes as needed
        for group_index, group_result in tqdm(pool.imap_unordered(process_group, range(num_groups)), total=num_groups):
            results.append((group_index, group_result))  # Append group_index along with result

    # Sort the results based on group_index to maintain order
    results.sort(key=lambda x: x[0])
    ordered_results = [result for _, result in results]

    # Flatten the ordered_results to get the final result
    result = [item for sublist in ordered_results for item in sublist]

    submit = pd.read_csv(sample)
    submit['mask_rle'] = result
    if not os.path.exists(csv_save_dir):
        os.makedirs(csv_save_dir)
    submit.to_csv(os.path.join(csv_save_dir, csv_name + '.csv'), index=False)