use crate::asap_embed_common::*;

use log::info;

use matrix_param::traits::Inference;
use matrix_param::traits::ParamIo;
use matrix_util::common_io::{extension, read_lines};
use matrix_util::dmatrix_rsvd::RSVD;
use matrix_util::traits::*;

use crate::asap_collapse_data::CollapsingOps;
use crate::asap_random_projection::RandProjOps;
use asap_data::sparse_io::*;
use asap_data::sparse_io_vector::*;

use candle_util::candle_data_loader::*;
use candle_util::candle_loss_functions as loss_func;
use candle_util::candle_model_encoder::*;
use candle_util::candle_model_topic::*;
use candle_util::candle_model_traits::*;
use candle_util::candle_vae_inference::*;

use clap::{Parser, ValueEnum};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use indicatif::ParallelProgressIterator;

