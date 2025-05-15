import os
from collections import Counter, defaultdict

def count_labels(labels_dir):
    counts = Counter()
    file_counts = defaultdict(int)
    for fname in os.listdir(labels_dir):
        if fname.endswith('.txt'):
            with open(os.path.join(labels_dir, fname), 'r') as f:
                for line in f:
                    if line.strip():
                        class_id = line.strip().split()[0]
                        counts[class_id] += 1
                        file_counts[class_id] += 1
    return counts, file_counts

train_labels = os.path.join('train', 'labels')
val_labels = os.path.join('val', 'labels')

train_counts, _ = count_labels(train_labels)
val_counts, _ = count_labels(val_labels)

def load_classes(classes_file):
    with open(classes_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

classes = load_classes('classes.txt')

print('Class counts in TRAIN:')
for cid, count in sorted(train_counts.items(), key=lambda x: int(x[0])):
    cname = classes[int(cid)] if int(cid) < len(classes) else f'class_{cid}'
    print(f'  {cid} ({cname}): {count}')

print('\nClass counts in VAL:')
for cid, count in sorted(val_counts.items(), key=lambda x: int(x[0])):
    cname = classes[int(cid)] if int(cid) < len(classes) else f'class_{cid}'
    print(f'  {cid} ({cname}): {count}')
