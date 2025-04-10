import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def extract_voice_features(audio_path):
    """Extract voice features from an audio segment."""
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Extract features
    # Mel-frequency cepstral coefficients (MFCCs)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)
    
    # Spectral features
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean()
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0].mean()
    
    # Pitch features
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitches[magnitudes > np.max(magnitudes)*0.1])
    
    # Combine features
    features = np.concatenate([
        mfcc_means,
        [spec_cent, spec_bw, pitch_mean]
    ])
    
    return features

def analyze_speakers():
    """Analyze and cluster speakers in the segments."""
    segments_dir = Path("data/olamide/speech_segments")
    output_dir = Path("data/olamide/speaker_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all WAV files
    wav_files = list(segments_dir.glob("*.wav"))
    if not wav_files:
        print("No segments found!")
        return
    
    print(f"\nAnalyzing {len(wav_files)} segments...")
    
    # Extract features for each segment
    features = []
    for wav_file in wav_files:
        try:
            feat = extract_voice_features(wav_file)
            features.append(feat)
            if len(features) % 10 == 0:
                print(f"Processed {len(features)} segments...")
        except Exception as e:
            print(f"Error processing {wav_file}: {str(e)}")
    
    # Convert to numpy array
    features = np.array(features)
    
    # Normalize features
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    
    # Cluster into 2 speakers using K-means
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(features)
    
    # Analyze clusters
    cluster_durations = {0: [], 1: []}
    for wav_file, label in zip(wav_files, labels):
        duration = float(wav_file.stem.split("_")[-1].replace("s", ""))
        cluster_durations[label].append(duration)
    
    # The cluster with longer average duration is likely Olamide (interviewee speaks longer)
    cluster_0_mean = np.mean(cluster_durations[0])
    cluster_1_mean = np.mean(cluster_durations[1])
    olamide_cluster = 0 if cluster_0_mean > cluster_1_mean else 1
    
    print(f"\nCluster 0 average duration: {cluster_0_mean:.2f}s")
    print(f"Cluster 1 average duration: {cluster_1_mean:.2f}s")
    print(f"Identified Olamide's cluster as: {olamide_cluster}")
    
    # Save results
    results = {
        'segments': [
            {
                'file': str(wav_file),
                'is_olamide': int(label) == olamide_cluster,
                'duration': float(wav_file.stem.split("_")[-1].replace("s", ""))
            }
            for wav_file, label in zip(wav_files, labels)
        ],
        'olamide_cluster': int(olamide_cluster),
        'cluster_stats': {
            'cluster_0_mean_duration': float(cluster_0_mean),
            'cluster_1_mean_duration': float(cluster_1_mean),
            'cluster_0_count': int(np.sum(labels == 0)),
            'cluster_1_count': int(np.sum(labels == 1))
        }
    }
    
    # Save analysis results
    output_path = output_dir / "speaker_analysis.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nAnalysis results saved to: {output_path}")
    
    # Create directory for Olamide's segments
    olamide_dir = Path("data/olamide/olamide_segments")
    olamide_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy Olamide's segments
    olamide_segments = []
    for wav_file, label in zip(wav_files, labels):
        if int(label) == olamide_cluster:
            output_path = olamide_dir / wav_file.name
            sf.write(output_path, sf.read(wav_file)[0], sf.read(wav_file)[1])
            olamide_segments.append(str(output_path))
    
    print(f"\nSaved {len(olamide_segments)} segments identified as Olamide's voice")
    
    # Plot feature visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(features[labels == 0, 0], features[labels == 0, 1], 
               c='blue', label='Cluster 0', alpha=0.6)
    plt.scatter(features[labels == 1, 0], features[labels == 1, 1], 
               c='red', label='Cluster 1', alpha=0.6)
    plt.title('Speaker Clustering Visualization')
    plt.xlabel('MFCC 1')
    plt.ylabel('MFCC 2')
    plt.legend()
    plt.savefig(output_dir / 'speaker_clusters.png')
    plt.close()

if __name__ == "__main__":
    analyze_speakers() 