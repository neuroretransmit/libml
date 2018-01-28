extern crate rand;

use rand::distributions::{IndependentSample, Range};

#[derive(Debug)]
pub struct Neuron {
    weights: Vec<f32>,
    output: f32,
    bias_idx: usize,
    delta: f32
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
            output: 0.0f32,
            bias_idx: num_inputs,
            delta: 0.0f32
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

    pub fn derivative(&self) -> f32 {
        1.0 - self.output.powi(2)
    }

    pub fn adjust_weight(&mut self, i: usize, x: f32) {
        self.weights[i] += x;
    }

    pub fn weight(&self, i: usize) -> f32 {
        self.weights[i]
    }

    pub fn set_delta(&mut self, delta: f32) {
        self.delta = delta;
    }

    pub fn delta(&self) -> f32 {
        self.delta
    }

    pub fn output(&self) -> f32 {
        self.output
    }

    pub fn bias_idx(&self) -> usize {
        self.bias_idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mnist_decoder::{ MNISTSequence, MNISTLabelFile, MNISTImageFile, MNIST_COLS, MNIST_ROWS };

    const INPUTS: [[f32; 2]; 4] = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ];

    macro_rules! float_to_bool {
        ($x:expr) => (if $x >= 0.5 { true } else { false })
    }

    macro_rules! to_f32_vec {
        ($x:expr) => ($x.iter().map(|&e| e as f32).collect())
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

    #[test]
    fn mnist_handwritten_database() {
        if let Some(mut train_images) = MNISTImageFile::new("data/mnist/train-images.idx3-ubyte") {
            match MNISTLabelFile::new("data/mnist/train-labels.idx1-ubyte") {
                Some(mut train_labels) => {
                    let mut neurons = Vec::new();

                    // Create a neuron for each label 0-9 with 784 inputs for each pixel in the 28x28 image
                    // Since this is greyscale, every pixel will be a value between 0-255
                    for _ in 0..10 {
                        neurons.push(Neuron::new(MNIST_ROWS * MNIST_COLS));
                    }

                    // Train neurons with 60k training samples
                    for _ in 0..train_images.num_items() {
                        let label = train_labels.next_item();
                        let inputs = train_images.next_item();
                        neurons[label as usize].train(&to_f32_vec!(inputs), label as f32, 0.2);
                    }

                    if let Some(mut test_images) = MNISTImageFile::new("data/mnist/t10k-images.idx3-ubyte") {
                        if let Some(mut test_labels) = MNISTLabelFile::new("data/mnist/t10k-labels.idx1-ubyte") {
                            let mut errors = 0;

                            for _ in 0..test_images.num_items() {
                                let test_label = test_labels.next_item();
                                let test_inputs = test_images.next_item();

                                for j in 0..neurons.len() {
                                    let output = neurons[j].fire(&to_f32_vec!(test_inputs));

                                    if j as u8 != test_label && float_to_bool!(output) {
                                        errors += 1;
                                    } else if j as u8 == test_label && !float_to_bool!(output) {
                                        errors += 1;
                                    }
                                }
                            }

                            let error_rate = (test_images.num_items() * neurons.len() as u32) as f32 / errors as f32;
                            eprintln!("MNIST error rate: {}", error_rate);
                            assert!(error_rate < 1.25);
                        }
                    }
                },
                _ => {
                    assert!(false);
                }
            }
        } else {
            assert!(false);
        }
    }
}
