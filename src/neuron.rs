extern crate rand;

use rand::distributions::{IndependentSample, Range};

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

    pub fn train(&mut self, inputs: &Vec<f32>, expected: f32, lrate: f32) {
        let output = self.fire(inputs);
        let local_err = expected - output;

        for j in 0..inputs.len() {
            self.weights[j] += lrate * local_err * inputs[j];
        }

        self.weights[self.bias_idx] += lrate * local_err;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const INPUTS: [[f32; 2]; 4] = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];

    macro_rules! float_to_bool {
        ($x:expr) => (if $x >= 0.5 { true } else { false })
    }

    #[test]
    fn fire() {
        let mut neuron: Neuron = Neuron::new(2);
        let inputs = vec![0.2, 0.3];
        let output = neuron.fire(&inputs);
        assert_eq!(output, inputs.iter()
            .zip(neuron.weights.iter() )
            .fold(neuron.weights[neuron.bias_idx], |sum, (a, b)| sum + a * b).tanh());
        assert_eq!(neuron.output, output);
    }

    #[test]
    fn train() {
        let mut neuron: Neuron = Neuron::new(2);
        let original_weights = neuron.weights.clone();
        neuron.train(&vec![0.2, 1.1], 1.0, 0.2);

        // Test weight change
        for i in 0..original_weights.len() {
            assert_ne!(original_weights[i], neuron.weights[i]);
        }
    }

    #[test]
    fn logical_or() {
        let mut neuron = Neuron::new(2);
        let expected = vec![
            0.0,
            1.0,
            1.0,
            1.0
        ];

        for _ in 0..2500 {
            for i in 0..INPUTS.len() {
                neuron.train(&INPUTS[i].to_vec(), expected[i], 0.2);
            }
        }

        for i in 0..INPUTS.len() {
            assert_eq!(float_to_bool!(neuron.fire(&INPUTS[i].to_vec())), float_to_bool!(expected[i]));
        }
    }

    #[test]
    fn logical_and() {
        let mut neuron = Neuron::new(2);
        let expected = vec![
            0.0,
            0.0,
            0.0,
            1.0
        ];

        for _ in 0..2500 {
            for i in 0..INPUTS.len() {
                neuron.train(&INPUTS[i].to_vec(), expected[i], 0.2);
            }
        }

        for i in 0..INPUTS.len() {
            assert_eq!(float_to_bool!(neuron.fire(&INPUTS[i].to_vec())), float_to_bool!(expected[i]));
        }
    }

    #[test]
    fn logical_nand() {
        let mut neuron = Neuron::new(2);
        let expected = vec![
            1.0,
            1.0,
            1.0,
            0.0
        ];

        for _ in 0..2500 {
            for i in 0..INPUTS.len() {
                neuron.train(&INPUTS[i].to_vec(), expected[i], 0.2);
            }
        }

        for i in 0..INPUTS.len() {
            assert_eq!(float_to_bool!(neuron.fire(&INPUTS[i].to_vec())), float_to_bool!(expected[i]));
        }
    }

    #[test]
    fn logical_nor() {
        let mut neuron = Neuron::new(2);
        let expected = vec![
            1.0,
            0.0,
            0.0,
            0.0
        ];

        for _ in 0..2500 {
            for i in 0..INPUTS.len() {
                neuron.train(&INPUTS[i].to_vec(), expected[i], 0.2);
            }
        }

        for i in 0..INPUTS.len() {
            assert_eq!(float_to_bool!(neuron.fire(&INPUTS[i].to_vec())), float_to_bool!(expected[i]));
        }
    }
}
