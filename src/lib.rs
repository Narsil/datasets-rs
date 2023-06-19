#![deny(missing_docs)]
//! This crates aims to emulate and be compatible with the
//! [datasets](https://github.com/huggingface/datasets/) python package.
//!
//! This crate limits itself to the actual parquet files and limits itself to making
//! the parquet files available easily
//!
//! At this time only a limited subset of the functionality is present, the goal is to add new
//! features over time
use arrow2::datatypes::Schema;
use hf_hub::{
    api::{Api, ApiError},
    Repo, RepoType,
};
use parquet2::{error::Error as ParquetError, metadata::FileMetaData, read::deserialize_metadata};
use reqwest::{
    header::{ToStrError, CONTENT_RANGE, RANGE},
    Error as ReqwestError,
};
use std::num::{ParseIntError, TryFromIntError};
use std::path::PathBuf;
use thiserror::Error;

/// The default trait to implement to get the simplest API
pub trait Dataset {
    /// The type of objects contained in the dataset
    type Item;

    /// The length of the dataset
    fn len(&self) -> usize;

    /// Get item at specific index. Should return `None` if and only if
    /// `index > dataset.len()`.
    fn get(&self, index: usize) -> Option<Self::Item>;
}

/// Generic structure to iterate over [`Dataset`].
pub struct DatasetIterator<'a, D> {
    dataset: &'a D,
    index: usize,
}

/// Iterate of the dataset in order
pub fn iter<'a, D: Dataset>(dataset: &'a D) -> DatasetIterator<'a, D> {
    DatasetIterator { dataset, index: 0 }
}

impl<'a, D: Dataset> Iterator for DatasetIterator<'a, D> {
    type Item = D::Item;

    fn next(&mut self) -> Option<Self::Item> {
        let object = self.dataset.get(self.index);
        self.index += 1;
        object
    }
}

/// The classic wikitext dataset
pub struct Wikitext103RawV1Test {
    local_parquet_files: Vec<(std::fs::File, Schema, FileMetaData)>,
}

impl Wikitext103RawV1Test {
    /// Download the local files for simple usage
    pub async fn new(api: &Api) -> Result<Self, DatasetError> {
        let remote_files = vec!["wikitext-103-raw-v1/wikitext-test.parquet".to_string()];
        let mut local_parquet_files = Vec::with_capacity(remote_files.len());
        let repo = Repo::with_revision(
            "wikitext".to_string(),
            RepoType::Dataset,
            "refs/convert/parquet".to_string(),
        );
        for rfilename in remote_files {
            let filename = api.download(&repo, &rfilename).await?;

            let mut local_file = std::fs::File::open(filename).unwrap();
            let metadata = parquet2::read::read_metadata(&mut local_file).unwrap();
            let schema = arrow2::io::parquet::read::infer_schema(&metadata).unwrap();
            local_parquet_files.push((local_file, schema, metadata))
        }
        Ok(Self {
            local_parquet_files,
        })
    }
}

/// Wikitext element
pub struct Item {
    /// The text of the page
    pub text: String,
}

impl Dataset for Wikitext103RawV1Test {
    type Item = Item;

    fn len(&self) -> usize {
        return 4358;
    }
    fn get(&self, mut index: usize) -> Option<Self::Item> {
        if index >= self.len() {
            return None;
        }
        // let start = std::time::Instant::now();
        for (local_file, schema, metadata) in &self.local_parquet_files {
            // println!("Open {:?}", start.elapsed());
            // we can read its metadata:
            // println!("Schema {:?}", start.elapsed());
            for row_group in &metadata.row_groups {
                let num_rows = row_group.num_rows();
                if num_rows > index {
                    let row_groups = vec![row_group.clone()];
                    // println!("RowGRoups {:?}", start.elapsed());
                    // TODO We need to skip some computation somehow
                    // use arrow2::io::parquet::read::indexes::{FilteredPage, Interval};
                    // let pages = Some(vec![vec![vec![vec![FilteredPage {
                    //     start: 0,
                    //     length: index,
                    //     selected_rows: vec![Interval {
                    //         start: index,
                    //         length: 0,
                    //     }],
                    //     num_rows: 1,
                    // }]]]]);
                    let mut reader = arrow2::io::parquet::read::FileReader::new(
                        local_file,
                        row_groups,
                        schema.clone(),
                        None,
                        None,
                        None,
                    );
                    // println!("FileReader {:?}", start.elapsed());
                    // Skip the chunk
                    let chunk = reader.next().unwrap();
                    // println!("Chunk1 {:?}", start.elapsed());
                    let chunk = chunk.unwrap();
                    // println!("Chunk {:?}", start.elapsed());
                    let array = chunk.arrays().iter().next().unwrap();
                    // println!("Array {:?}", start.elapsed());
                    let item = array;
                    // println!("Before downcast {:?}", start.elapsed());
                    let text: String = item
                        .as_any()
                        .downcast_ref::<arrow2::array::Utf8Array<i32>>()
                        .unwrap()
                        .value(0)
                        .to_string();
                    // println!("After downcast {:?}", start.elapsed());
                    // println!("Final {:?}", start.elapsed());
                    return Some(Item { text });
                    // let chunk = reader.skip(index).next().unwrap().unwrap();
                    // println!("Text {text:?}");
                    // return Some(Item { text });
                } else {
                    index -= num_rows;
                }
            }
        }
        return None;
    }
}

/// When fetching parquet metadata, we fetch more than the last 8 bytes
/// in order to optimize round trips if the metadata is small enough
/// A very minimal metadata is 2kB, so 100kB covers most small datasets while
/// it should be minimal overhead of modern connections
const PARQUET_METADATA_MIN_SIZE: usize = 100_00;

/// Error type for datasets
#[derive(Debug, Error)]
pub enum DatasetError {
    /// The error comes from api usage.
    #[error("api: {0}")]
    ApiError(#[from] ApiError),

    /// We expected the magic number in the parquet file but didn't see it.
    #[error("Invalid parquet magic number")]
    InvalidParquetMagic,

    /// parquet error
    #[error("ParquetError: {0}")]
    ParquetError(#[from] ParquetError),

    /// The header value is not valid utf-8
    #[error("header value is not a string")]
    ToStr(#[from] ToStrError),

    /// Error in the request
    #[error("request error: {0}")]
    RequestError(#[from] ReqwestError),

    /// Error parsing some range value
    #[error("Cannot parse int: {0}")]
    ParseIntError(#[from] ParseIntError),

    /// Error converting some ints
    #[error("Cannot convert int: {0}")]
    TryFromIntError(#[from] TryFromIntError),
}

const PARQUET_MAGIC: [u8; 4] = [b'P', b'A', b'R', b'1'];

/// The core struct used to interact with a dataset
pub struct HubDataset {
    api: Api,
    repo: Repo,
}

impl HubDataset {
    /// The id is the canonical hub name: you can look for names on the [hub](https://huggingface.co/datasets)
    pub fn from_id(id: String) -> Result<Self, ApiError> {
        let api = Api::new()?;
        Ok(Self::new(api, id))
    }

    /// Create the dataset if you already have an [`hf_hub::api::Api`] at hand.
    pub fn new(api: Api, id: String) -> Self {
        let repo = Repo::with_revision(id, RepoType::Dataset, "refs/convert/parquet".to_string());
        Self { api, repo }
    }

    /// Lists the available parquet files on the remote.
    pub async fn remote_files(&self) -> Result<Vec<String>, ApiError> {
        let info = self.api.info(&self.repo).await?;
        let mut filenames = Vec::with_capacity(info.siblings.len());
        for sibling in info.siblings {
            if sibling.rfilename.ends_with(".parquet") {
                filenames.push(sibling.rfilename);
            }
        }
        Ok(filenames)
    }

    /// Lists the available parquet files on the remote.
    pub async fn parquet_metadata(
        &self,
        remote_filename: &str,
    ) -> Result<FileMetaData, DatasetError> {
        let url = self.api.url(&self.repo, remote_filename);
        let response = self
            .api
            .client()
            .get(&url)
            .header(RANGE, "bytes=0-0")
            .send()
            .await?;
        let headers = response.headers();
        let content_range = headers
            .get(CONTENT_RANGE)
            .ok_or(ApiError::MissingHeader(CONTENT_RANGE))?
            .to_str()?;

        let size: usize = content_range
            .split('/')
            .last()
            .ok_or(ApiError::InvalidHeader(CONTENT_RANGE))?
            .parse()?;

        let stop = size;
        let min_size = PARQUET_METADATA_MIN_SIZE;
        let start = if size < min_size { 0 } else { stop - min_size };
        let response = self
            .api
            .client()
            .get(&url)
            .header(RANGE, format!("bytes={start}-{stop}"))
            .send()
            .await?;
        let buffer = response.bytes().await?;
        let len = buffer.len();
        if buffer[len - 4..] != PARQUET_MAGIC {
            return Err(DatasetError::InvalidParquetMagic);
        }
        let metadata_len = i32::from_le_bytes(buffer[len - 8..len - 4].try_into().unwrap());
        let metadata_len: usize = metadata_len.try_into()?;

        // a highly nested but sparse struct could result in many allocations
        let max_size = metadata_len * 2 + 1024;
        let metadata = if metadata_len < len - 8 {
            // Happy path, we already fetched the metadata
            deserialize_metadata(&buffer[len - metadata_len - 8..len - 8], max_size)?
        } else {
            // Unhappy path, let's fetch the metadata.
            // Fetching everything, so we don't have to copy anything
            let start = size - 8 - metadata_len;
            let stop = size - 8;
            let response = self
                .api
                .client()
                .get(url)
                .header(RANGE, format!("bytes={start}-{stop}"))
                .send()
                .await?;

            deserialize_metadata(&response.bytes().await?[..], max_size)?
        };

        Ok(metadata)
        // for column in schema.columns() {
        //     println!("Col {column:?}");
        // }

        // todo!();
        // // todo!("Meta {schema:#?}");
    }

    /// Lists the available parquet files on the local cache directory after downloading them.
    pub async fn download(&self) -> Result<Vec<PathBuf>, ApiError> {
        let rfilenames = self.remote_files().await?;
        let mut filenames = Vec::with_capacity(rfilenames.len());
        for rfilename in rfilenames {
            let filename = self.api.download(&self.repo, &rfilename).await?;
            filenames.push(filename);
        }
        Ok(filenames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hf_hub::api::ApiBuilder;

    #[tokio::test]
    async fn get_wikitext_parquet_files() {
        let dataset = HubDataset::from_id("wikitext".to_string()).unwrap();
        let remote_filenames = dataset.remote_files().await.unwrap();
        assert_eq!(
            remote_filenames,
            vec![
                "wikitext-103-raw-v1/wikitext-test.parquet",
                "wikitext-103-raw-v1/wikitext-train-00000-of-00002.parquet",
                "wikitext-103-raw-v1/wikitext-train-00001-of-00002.parquet",
                "wikitext-103-raw-v1/wikitext-validation.parquet",
                "wikitext-103-v1/wikitext-test.parquet",
                "wikitext-103-v1/wikitext-train-00000-of-00002.parquet",
                "wikitext-103-v1/wikitext-train-00001-of-00002.parquet",
                "wikitext-103-v1/wikitext-validation.parquet",
                "wikitext-2-raw-v1/wikitext-test.parquet",
                "wikitext-2-raw-v1/wikitext-train.parquet",
                "wikitext-2-raw-v1/wikitext-validation.parquet",
                "wikitext-2-v1/wikitext-test.parquet",
                "wikitext-2-v1/wikitext-train.parquet",
                "wikitext-2-v1/wikitext-validation.parquet"
            ]
        );
    }

    #[tokio::test]
    async fn get_wikitext_simple_dataset() {
        let api = ApiBuilder::new().with_progress(false).build().unwrap();
        let dataset = Wikitext103RawV1Test::new(&api).await.unwrap();
        assert_eq!(dataset.len(), 4358);
        let start = std::time::Instant::now();
        let count = iter(&dataset).count();
        println!("Took {:?}", start.elapsed());
        assert_eq!(count, 4358);
    }
}
