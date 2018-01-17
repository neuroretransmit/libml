extern crate rand;

use rand::distributions::{IndependentSample, Range};
use tdata::{TData};

#[derive(Debug)]
struct Neuron {
    weights: Vec<f32>,
    delta: f32,
    output: f32,
    bias_idx: usize,
    expected_idx: usize
}

impl Neuron {
    pub fn new(num_inputs: usize) -> Neuron {
        let between = Range::new(-1f32, 1.);
        let mut rng = rand::thread_rng();
        let mut weights = Vec::new();
        
        for _ in 0..num_inputs + 1 /* bias */ {
            weights.push(between.ind_sample(&mut rng));
        }

        Neuron {
            weights,
            delta: 0.0f32,
            output: 0.0f32,
            bias_idx: num_inputs,
            expected_idx: num_inputs
        }
    }

    pub fn fire(&mut self, inputs: &Vec<f32>) -> f32 {
        self.output = inputs.iter()
            .zip(self.weights.iter() )
            .fold(self.weights[self.bias_idx], |sum, (a, b)| sum + a * b).tanh();
        self.output
    }

    pub fn train(&mut self, data: &TData, epochs: usize, lrate: f32) {
        for _ in 0..epochs {
            for sample in data.samples() {
                let output = self.fire(sample.inputs());
                let local_err = sample.outputs()[0] - output;

                for j in 0..sample.inputs().len() {
                    self.weights[j] += lrate * local_err * sample.inputs()[j];
                }
                
                self.weights[self.bias_idx] += lrate * local_err;
            }
        }
    }

    pub fn weights(&self) -> &Vec<f32> {
        &self.weights
    }

    pub fn delta(&self) -> &f32 {
        &self.delta
    }

    pub fn delta_mut(&mut self) -> &f32 {
        &self.delta
    }

    pub fn output(&self) -> &f32 {
        &self.output
    }
}

#[cfg(test)]
mod tests {
    mod neuron {
        use neuron::{Neuron};
        use tdata::{TData};
        
        #[test]
        fn new() {
            let neuron = Neuron::new(2);
            let weights = neuron.weights();
            assert_eq!(weights.len(), 3);
            
            for w in weights {
                assert_ne!(*w, 0.0);
            }

            assert_eq!(*neuron.delta(), 0.0f32);
            assert_eq!(*neuron.output(), 0.0f32);
        }

        #[test]
        fn train() {
            let mut neuron: Neuron = Neuron::new(2);
            let data: TData = TData::new(2, 1, "data/or.csv").unwrap();
            let original_weights = neuron.weights().clone();
            neuron.train(&data, 2000, 0.2);
            
            // Test weight change
            for i in 0..original_weights.len() {
                assert_ne!(original_weights[i], neuron.weights()[i]);
            }
           
            // Test OR
            for sample in data.samples() {
                assert_eq!(if neuron.fire(sample.inputs()) >= 0.5 { 1.0 } else { 0.0 }, sample.outputs()[0]);
            }

            let mut neuron: Neuron = Neuron::new(2);
            let data: TData = TData::new(2, 1, "data/and.csv").unwrap();
            neuron.train(&data, 2000, 0.2);

            // Test AND
            for sample in data.samples() {
                assert_eq!(if neuron.fire(sample.inputs()) >= 0.5 { 1.0 } else { 0.0 }, sample.outputs()[0]);
            }

            let mut neuron: Neuron = Neuron::new(2);
            let data: TData = TData::new(2, 1, "data/nand.csv").unwrap();
            neuron.train(&data, 2000, 0.2);

            // Test NAND
            for sample in data.samples() {
                assert_eq!(if neuron.fire(sample.inputs()) >= 0.5 { 1.0 } else { 0.0 }, sample.outputs()[0]);
            }

            let mut neuron: Neuron = Neuron::new(2);
            let data: TData = TData::new(2, 1, "data/nor.csv").unwrap();
            neuron.train(&data, 2000, 0.2);

            // Test NOR
            for sample in data.samples() {
                assert_eq!(if neuron.fire(sample.inputs()) >= 0.5 { 1.0 } else { 0.0 }, sample.outputs()[0]);
            }
        }
    }
}
