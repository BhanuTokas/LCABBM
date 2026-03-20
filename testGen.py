from data import get_dataset
from types import SimpleNamespace
from conceptEval import ConceptDrifter

args = SimpleNamespace(out_dir = "outputs/test1/", dataset = "cub", batch_size=32, num_workers=1)

train_d, test_d, idx_to_classes, classes = get_dataset(args)

c = ConceptDrifter()

text_positive = "blue"
text_negative = "red"
embeddings = c.clip_interface.getTextEmbedding(
    [text_positive, text_negative]
)
z_pos = embeddings[0:1]
z_neg = embeddings[1:2]

count=0
for image, label in test_d.dataset:
    image = image.resize((512,512))
    img1, img2 = c.perturbImagePoints(image, z_pos, z_neg, delta=0.1)
    img1.show()
    img2.show()
    count+=1
    if(count>2):
        break
'''
'''