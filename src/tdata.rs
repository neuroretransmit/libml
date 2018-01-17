extern crate csv;

use std::error::Error;
use std::fs::File;
use csv::{ReaderBuilder};

#[derive(Debug)]
pub struct Sample {
    inputs: Vec<f32>,
    outputs: Vec<f32>
}

impl Sample {
    pub fn new(inputs: Vec<f32>, outputs: Vec<f32>) -> Sample {
        Sample {
            inputs,
            outputs
        }
    }

    pub fn inputs(&self) -> &Vec<f32> {
        &self.inputs
    }

    pub fn outputs(&self) -> &Vec<f32> {
        &self.outputs
    }
}

pub struct TData {
    samples: Vec<Sample>
}

impl TData {
    pub fn new(num_inputs: usize, num_outputs: usize, fname: &'static str) -> Result<TData, Box<Error>> {
        let mut samples: Vec<Sample> = Vec::new();
        let file = File::open(fname)?;
        let mut rdr = ReaderBuilder::new().has_headers(false).from_reader(file);
        
        for result in rdr.records() {
            let mut inputs: Vec<f32> = Vec::new();
            let mut outputs: Vec<f32> = Vec::new();

            let record = result?;

            for i in 0..num_inputs {
                inputs.push(record.get(i).unwrap().parse::<f32>().unwrap());

            }

            for i in num_inputs..num_inputs + num_outputs {
                outputs.push(record.get(i).unwrap().parse::<f32>().unwrap());
            }

            samples.push(Sample::new(inputs, outputs));
        }
        Ok(
            TData {
                samples
            }
        )
    }

    pub fn samples(&self) -> &Vec<Sample> {
        &self.samples
    }
}

#[cfg(test)]
mod tests {
    mod tdata {
        use tdata::{TData};

        #[test]
        fn new() {
            let data = TData::new(2, 1, "data/or.csv");
            assert_eq!(data.unwrap().samples.len(), 4);
        }
    }
}
