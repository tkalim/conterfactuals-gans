from torch.utils.data import DataLoader, dataloader

from src.infer_latent_factors import LATENT_FACTORS_DIR

from .data.datasets import ConceptsDataset
from .tcav.cav import CAV

ATTRIBUTES_CSV_FILE = "./data/celeba/list_attr_celeba.txt"
LATENT_FACTORS_DIR = "./data/generated/celeba-latent-factors"
BOTTLENECK = "latent-factors"


def get_activations(dataloader, concept):
    """Returns the latent factors in CAV-compatible format

    Args:
        dataloader (DataLoader): DataLoader for the latent-factors
            dataset
        concept (str): name of the concept contained in the dataset

    Returns:
        dict: is a dictionary of activations. In particular, acts takes for of
                {'concept1':{'bottleneck name1':[...act array...],
                             'bottleneck name2':[...act array...],...
                 'concept2':{'bottleneck name1':[...act array...],
    """
    activation_concept_array = []
    activation_random_array = []
    for ind, batch in enumerate(dataloader):
        for sample in batch:
            latent_factor = sample["latent_factor"].numpy()
            label = sample["label"]
            if label == 1:
                activation_concept_array.append(latent_factor)
            else:
                activation_random_array.append(latent_factor)

    output_dict = {
        concept: {BOTTLENECK: activation_concept_array},
        "random": {BOTTLENECK: activation_random_array},
    }
    return output_dict


if __name__ == "__main__":
    dataset = ConceptsDataset(
        attributes_csv_file=ATTRIBUTES_CSV_FILE,
        root_dir=LATENT_FACTORS_DIR,
        concept="Young",
        n_samples=500,
    )
    concept = dataset.concept
    dataloader = DataLoader(dataset=dataset, batch_size=dataset.__len__())
    acts = get_activations(dataloader=dataloader, concept=concept)
    cav = CAV(concepts=(concept, "random"), bottleneck=BOTTLENECK)
    cav.train(acts=acts)
    face_attribute_vector = cav.get_direction()
