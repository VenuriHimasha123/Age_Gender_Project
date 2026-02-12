import pandas as pd
import os

# Path to your unzipped UTKFace images
path = "data/UTKFace/"

if not os.path.exists(path):
    print(f"❌ Error: Folder not found at {path}")
else:
    files = os.listdir(path)
    data = []

    print(f"🔄 Processing {len(files)} images...")

    for file in files:
        # Format: [age]_[gender]_[race]_[date].jpg
        parts = file.split('_')
        
        # We ensure it has at least 4 parts to get Age, Gender, and Race
        if len(parts) >= 4:
            try:
                age = int(parts[0])
                gender = int(parts[1])
                race = int(parts[2])
                
                data.append({
                    'image': file,
                    'age': age,
                    'gender': gender,
                    'race': race
                })
            except (ValueError, IndexError):
                continue

    # Create the master spreadsheet
    df = pd.DataFrame(data)
    
    # Save it
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/training_data.csv", index=False)
    
    print("--- DATASET SUMMARY ---")
    print(f"✅ Total Images Mapped: {len(df)}")
    print(f"📊 Age Range: {df['age'].min()} - {df['age'].max()}")
    print(f"🚻 Gender Count: Male={len(df[df['gender']==0])}, Female={len(df[df['gender']==1])}")
    print("-----------------------")