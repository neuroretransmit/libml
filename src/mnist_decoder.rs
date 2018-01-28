use std::io::prelude::*;
use std::io::Cursor;
use std::fs::File;
use byteorder::{BigEndian, ReadBytesExt};

pub const MNIST_ROWS: usize = 28;
pub const MNIST_COLS: usize = 28;
const MNIST_IMAGE_MAGIC: u32 = 2051;
const MNIST_LABEL_MAGIC: u32 = 2049;

pub trait MNISTSequence<T, I> {
    fn new(fname: &'static str) -> Option<T>;
    fn next_item(&mut self) -> I;
    fn magic(&self) -> u32;
    fn num_items(&self) -> u32;
}

pub struct MNISTImageFile {
    file: File,
    magic_number: u32,
    num_items: u32,
    num_rows: u32,
    num_columns: u32
}

pub struct MNISTLabelFile {
    file: File,
    magic_number: u32,
    num_items: u32
}

impl MNISTSequence<MNISTImageFile, Vec<u8>> for MNISTImageFile {
    fn new(fname: &'static str) -> Option<MNISTImageFile> {
        match File::open(fname) {
            Ok(mut file) => {
                let mut magic_number: u32;
                let mut num_items: u32;
                let mut num_rows: u32;
                let mut num_columns: u32;
                let mut buf = [0u8; 4];
                let _ = file.read(&mut buf);
                let mut rdr = Cursor::new(buf);
                magic_number = rdr.read_u32::<BigEndian>().unwrap();
                assert!(magic_number == MNIST_IMAGE_MAGIC);
                buf = [0u8; 4];
                let _ = file.read(&mut buf);
                rdr = Cursor::new(buf);
                num_items = rdr.read_u32::<BigEndian>().unwrap();
                buf = [0u8; 4];
                let _ = file.read(&mut buf);
                rdr = Cursor::new(buf);
                num_rows = rdr.read_u32::<BigEndian>().unwrap();
                buf = [0u8; 4];
                let _ = file.read(&mut buf);
                rdr = Cursor::new(buf);
                num_columns = rdr.read_u32::<BigEndian>().unwrap();

                Some(
                    MNISTImageFile {
                        file,
                        magic_number,
                        num_items,
                        num_rows,
                        num_columns
                    }
                )
            },
            _ => {
                None
            }
        }
    }

    fn next_item(&mut self) -> Vec<u8> {
        let mut buf = [0u8;  MNIST_ROWS * MNIST_COLS];
        let _ = self.file.read(&mut buf);
        return buf.to_vec();
    }

    fn magic(&self) -> u32 {
        self.magic_number
    }

    fn num_items(&self) -> u32 {
        self.num_items
    }
}

impl MNISTSequence<MNISTLabelFile, u8> for MNISTLabelFile {
    fn new(fname: &'static str) -> Option<MNISTLabelFile> {
        if let Ok(mut file) = File::open(fname) {
            let mut magic_number: u32;
            let mut num_items: u32;
            let mut buf = [0u8; 4];
            let _ = file.read(&mut buf);
            let mut rdr = Cursor::new(buf);
            magic_number =  rdr.read_u32::<BigEndian>().unwrap();
            assert!(magic_number == MNIST_LABEL_MAGIC);
            buf = [0u8; 4];
            let _ = file.read(&mut buf);
            rdr = Cursor::new(buf);
            num_items = rdr.read_u32::<BigEndian>().unwrap();

            Some(
                MNISTLabelFile {
                    file,
                    magic_number,
                    num_items
                }
            )
        } else {
            None
        }
    }

    fn next_item(&mut self) -> u8 {
        let mut buf = [0u8; 1];
        let _ = self.file.read(&mut buf);
        return buf[0];
    }

    fn magic(&self) -> u32 {
        self.magic_number
    }

    fn num_items(&self) -> u32 {
        self.num_items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const IMAGE_FILE: &'static str = "data/mnist/train-images.idx3-ubyte";
    const LABEL_FILE: &'static str = "data/mnist/train-labels.idx1-ubyte";

    #[test]
    pub fn image_file_new() {
        match MNISTImageFile::new(IMAGE_FILE) {
            Some(x) => {
                assert_eq!(2051, x.magic_number);
                assert_eq!(60000, x.num_items);
                assert_eq!(28, x.num_rows);
                assert_eq!(28, x.num_columns);
            }

            None => {
                assert!(false);
            }
        }
    }

    #[test]
    pub fn label_file_new() {
        match MNISTLabelFile::new(LABEL_FILE) {
            Some(x) => {
                assert_eq!(2049, x.magic_number);
                assert_eq!(60000, x.num_items);
            }

            None => {
                assert!(false);
            }
        }
    }

    #[test]
    pub fn image_file_next_item() {
        if let Some(mut mnist) = MNISTImageFile::new(IMAGE_FILE) {
            let image = mnist.next_item();

            // There are 9 rows filled with zeroes after the
            // preamble that gets read in MNISTFile::new()
            for i in 0..16*9 {
                assert_eq!(0x00, image[i]);
            }

            // The first line of hex values with data is this
            let expected: [u8; 16] = [ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x12, 0x12, 0x12, 0x7E, 0x88, 0xAF, 0x1A ];
            for i in 0..expected.len() {
                assert_eq!(expected[i], image[16*9 + i]);
            }
        }
    }

    #[test]
    pub fn label_file_next_item() {
        if let Some(mut mnist) = MNISTLabelFile::new(LABEL_FILE) {
            let expected: [u8; 16] = [ 0x05, 0x00, 0x04, 0x01, 0x09, 0x02, 0x01, 0x03, 0x01, 0x04, 0x03, 0x05, 0x03, 0x06, 0x01, 0x07 ];

            for e in expected.iter() {
                assert_eq!(*e, mnist.next_item());
            }
        }
    }
}