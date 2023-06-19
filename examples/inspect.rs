use datasets::HubDataset;
use parquet2::{
    metadata::SchemaDescriptor,
    schema::types::{PhysicalType, PrimitiveLogicalType, PrimitiveType},
};
use std::collections::{HashMap, HashSet};

fn dtype_from_logical_type(dtype: &PrimitiveType) -> String {
    match (dtype.logical_type, dtype.physical_type) {
        (Some(PrimitiveLogicalType::String), _) => "String".to_string(),
        (_, PhysicalType::Boolean) => "bool".to_string(),
        (_, PhysicalType::Int32) => "i32".to_string(),
        (_, PhysicalType::Int64) => "i64".to_string(),
        (_, PhysicalType::Float) => "f32".to_string(),
        (_, PhysicalType::Double) => "f64".to_string(),
        dt => panic!("Unhandled: {dt:?}"),
    }
}

fn get_item_string(metadata: &SchemaDescriptor) -> String {
    let mut string = String::new();
    string.push_str("pub struct Item{\n");
    for column in metadata.columns() {
        let name = &column.descriptor.primitive_type.field_info.name;
        let dtype = dtype_from_logical_type(&column.descriptor.primitive_type);
        string.push_str(&format!("    {name}: {dtype},\n"));
    }
    string.push_str("}");
    string
}

#[tokio::main]
async fn main() {
    let id = std::env::args()
        .nth(1)
        .expect("Give a specific dataset_id, choose from https://huggingface.co/datasets");
    let dataset = HubDataset::from_id(id).unwrap();

    let remote_files = dataset.remote_files().await.unwrap();
    let mut configs = HashMap::new();

    for remote_file in &remote_files {
        let mut fsplits = remote_file.split('/');
        let config = fsplits.next().unwrap();
        let rest = fsplits.next().unwrap();

        let mut subsplits = rest.split('.').next().unwrap().split('-');
        // dataset_id
        let _ = subsplits.next().unwrap();
        let split_name = subsplits.next().unwrap();
        assert_eq!(fsplits.next(), None);

        let metadata = dataset.parquet_metadata(remote_file).await.unwrap();

        let item_string = get_item_string(metadata.schema());
        configs
            .entry((config, split_name))
            .or_insert(HashSet::new())
            .insert((item_string, metadata.num_rows, remote_file));
    }
    for ((config, split_name), set) in configs {
        println!("");
        println!("--{config}/{split_name}.rs--");
        for (item_string, num_rows, remote_file) in set {
            println!("{}", item_string);
            println!("Filename: {remote_file}");
            println!("Count: {}", num_rows);
        }
    }
}
