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
    use neuron::{Neuron};

    #[test]
    fn new() {
        let neuron = Neuron::new(2);
        assert_eq!(neuron.weights.len(), 3);

        for w in neuron.weights {
            assert_ne!(w, 0.0);
        }

        assert_eq!(neuron.delta, 0.0f32);
        assert_eq!(neuron.output, 0.0f32);
    }

    #[test]
    fn fire() {
        let mut neuron: Neuron = Neuron::new(2);
        let inputs = vec![0.2, 0.3];
        assert_eq!(neuron.fire(&inputs), inputs.iter()
            .zip(neuron.weights.iter() )
            .fold(neuron.weights[neuron.weights.len() - 1], |sum, (a, b)| sum + a * b).tanh());
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
}
