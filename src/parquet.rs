use crate::Dataset;
use hf_hub::{
    api::{Api, ApiError},
    Repo, RepoType,
};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::errors::ParquetError;
use parquet::file::reader::SerializedFileReader;
use parquet::record::Row;
use std::fs::File;
use std::path::{Path, PathBuf};
use thiserror::Error;

pub struct ParquetDataset {
    rows: Vec<SerializedFileReader<File>>,
}

#[derive(Debug, Error)]
pub enum ParquetDatasetError {
    #[error("Api Error: {0}")]
    ApiError(#[from] ApiError),
    #[error("Parquet Error: {0}")]
    ParquetError(#[from] ParquetError),
    #[error("IO Error: {0}")]
    IoError(#[from] std::io::Error),
}

impl ParquetDataset {
    pub fn new(paths: &[&PathBuf]) -> Result<Self, ParquetDatasetError> {
        let rows: Result<Vec<SerializedFileReader<std::fs::File>>, ParquetDatasetError> = paths
            .into_iter()
            .map(
                |p| -> Result<SerializedFileReader<std::fs::File>, ParquetDatasetError> {
                    Ok(SerializedFileReader::try_from(std::fs::File::open(p)?)?)
                },
            )
            .collect();
        let rows: Vec<SerializedFileReader<std::fs::File>> = rows?;

        Ok(Self { rows })
    }

    pub async fn from_model_id(dataset_id: &str) -> Result<Self, ParquetDatasetError> {
        let api = Api::new()?;
        let repo = Repo::with_revision(
            "wikitext".to_string(),
            RepoType::Dataset,
            "refs/convert/parquet".to_string(),
        );
        let filename = api
            .download(&repo, "wikitext-103-v1/wikitext-test.parquet")
            .await?;
        Self::new(&[&filename])
    }
}

impl Dataset for ParquetDataset {
    type Item = Row;

    fn get(&self, index: usize) -> Option<Self::Item> {
        for reader in &self.rows {
            // let count = reader.into_iter().clone().count();
            let iter = reader.into_iter();
            // if count >= index {
            return iter.skip(index).next();
            // } else {
            //     index -= count;
            // }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wikitext() {
        let dataset = ParquetDataset::from_model_id("wikitext").await.unwrap();
        for item in dataset.iter() {
            println!("Item {item:?}");
        }
    }
}
