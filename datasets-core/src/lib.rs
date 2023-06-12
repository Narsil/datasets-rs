pub trait Dataset {
    pub type Item;
    pub fn download() -> Result<(), DatasetError>;
    pub fn len(&self) -> usize;
    pub fn get(&self, index: usize) -> Result<Item>;
}

pub struct HfDataset {
    dataset_id: String,
}
