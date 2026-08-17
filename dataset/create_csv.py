import pandas as pd
import random
import argparse
from pathlib import Path
import os

def load_original_csv(csv_path):
    #df = pd.read_csv(csv_path)
    df = pd.read_csv(csv_path, header=None, names=['filename', 'label', 'category'])
    return df

def categorize_label(label):
    speech_labels = ["female speech, woman speaking", 
                     "male speech, man speaking", 
                     "child speech, kid speaking",
                     "female singing",
                     "male singing",
                     "child singing"]
    
    music_labels = ["playing piano", 
                    "playing clarinet", 
                    "playing accordion"]
    
    others_labels = ["dog barking", "thunder"]
    
    if label in speech_labels:
        return "speech"
    elif label in music_labels:
        return "music"
    elif label in others_labels:
        return "others"
    else:
        return label

def generate_mixed_csv(original_df, output_csv, num_samples, num_sources):
    target_labels = [
        "female speech, woman speaking",
        "male speech, man speaking", 
        "child speech, kid speaking",
        "female singing",
        "male singing",
        "child singing",
        "playing piano",
        "playing clarinet",
        "playing accordion",
        "dog barking", 
        "thunder"
    ]
    
    target_df = original_df[original_df['label'].isin(target_labels)]
    
    print(original_df['label'].unique()[:20])
    if target_df.empty:
        print("错误: 未找到目标标签的数据")
        return

    target_df['category'] = target_df['label'].apply(categorize_label)
    output_data = []
    
    for i in range(num_samples):
        while True:
            selected_samples = target_df.sample(n=num_sources, replace=True)
            
            current_categories = selected_samples['category'].tolist()
            
            if len(set(current_categories)) > 1:
                break

        row_data = []
        
        for _, sample in selected_samples.iterrows():
            filename = sample['filename']
            #print(filename)
            #exit()
            if os.path.exists(filename):
                #print(filename)
                #exit()
                audio_path = filename
            else:
                wav_filename = filename.replace('.mp4', '.wav')
                audio_path = f"/work107/luoxiaoxue/data/VGGSound/audios_16k/{wav_filename}"
            category = sample['category']
            
            snr_range = [-3.0, 3.0]
            snr_db = random.uniform(snr_range[0], snr_range[1])
            row_data.extend([audio_path, category, snr_db])
        
        output_data.append(row_data)
    column_names = []
    
    for i in range(1, num_sources + 1):
        column_names.extend([f's{i}_path', f's{i}_label', f's{i}_snr'])
    
    output_df = pd.DataFrame(output_data, columns=column_names)
    
    output_df.to_csv(output_csv, index=False)
    print(f"已生成 {num_samples} 条混合音频数据到 {output_csv}")
    print(f"数据形状: {output_df.shape}")

def main():
    parser = argparse.ArgumentParser(description='生成指定混合声源数量的CSV文件')
    parser.add_argument('--input_csv', type=str, required=True,
                       help='原始CSV文件路径')
    parser.add_argument('--output_csv', type=str, required=True,
                       help='输出CSV文件路径')
    parser.add_argument('--type', type=str, choices=['train', 'valid', 'test'], required=True,
                       help='生成的数据集类型: train或valid')
    parser.add_argument('--num_sources', type=int, choices=[2, 3, 4, 5], required=True,
                       help='混合声源数量: 2, 3, 4 或 5')
    
    args = parser.parse_args()
    
    # 根据数据类型确定样本数量
    if args.type == 'train':
        num_samples = 20000
    elif args.type == 'valid':
        num_samples = 5000
    elif args.type == 'test':
        num_samples = 3000
    else:
        print(f"错误的数据类型: {args.type}")
        return
    
    print(f"正在加载原始CSV文件: {args.input_csv}")
    original_df = load_original_csv(args.input_csv)
    
    print(f"原始数据形状: {original_df.shape}")
    print(f"开始生成{args.type}数据，样本数: {num_samples}")
    print(f"original_df:\n{original_df.head()}\n...")
    
    generate_mixed_csv(
        original_df=original_df,
        output_csv=args.output_csv,
        num_samples=num_samples,
        num_sources=args.num_sources
    )

if __name__ == "__main__":
    main()
