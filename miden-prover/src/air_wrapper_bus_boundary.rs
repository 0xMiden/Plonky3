
use std::marker::PhantomData;

use miden_air::{BusType, MidenAir, MidenAirBuilder};
use p3_field::{PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::Matrix;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;

use crate::{StarkGenericConfig, Val};

pub struct AirWithBoundaryConstraints<'a, SC, A>
where
    SC: StarkGenericConfig + std::marker::Sync,
    A: MidenAir<Val<SC>, SC::Challenge>,
    Val<SC>: TwoAdicField,
{
    pub inner: &'a A,
    pub phantom: PhantomData<SC>,
}

impl<'a, SC, A> MidenAir<Val<SC>, SC::Challenge> for AirWithBoundaryConstraints<'a, SC, A>
where
    SC: StarkGenericConfig + std::marker::Sync,
    A: MidenAir<Val<SC>, SC::Challenge>,
    Val<SC>: TwoAdicField,
{
    fn width(&self) -> usize {
        self.inner.width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<Val<SC>>> {
        self.inner.preprocessed_trace()
    }

    fn num_public_values(&self) -> usize {
        self.inner.num_public_values()
    }

    fn periodic_table(&self) -> Vec<Vec<Val<SC>>> {
        self.inner.periodic_table()
    }

    fn num_randomness(&self) -> usize {
        self.inner.num_randomness()
    }

    fn aux_width(&self) -> usize {
        self.inner.aux_width()
    }

    /// Types of buses
    fn bus_types(&self) -> Vec<BusType> {
        self.inner.bus_types()
    }

    fn build_aux_trace(
        &self,
        _main: &RowMajorMatrix<Val<SC>>,
        _challenges: &[SC::Challenge],
    ) -> Option<RowMajorMatrix<Val<SC>>> {
        self.inner.build_aux_trace(_main, _challenges)
    }

    fn eval<AB: MidenAirBuilder<F = Val<SC>>>(&self, builder: &mut AB) {
        // First, apply the inner AIR's constraints
        self.inner.eval(builder);
        
        if self.inner.num_randomness() > 0 {
            // Then, apply any additional boundary constraints as needed
            let aux = builder.permutation();
            let aux_current = aux.row_slice(0).unwrap();
            let aux_bus_boundary_values = builder.aux_bus_boundary_values().to_vec();

            for (idx, bus_type) in self.inner.bus_types().into_iter().enumerate() {
                match bus_type {
                    BusType::Multiset => {
                        builder.when_first_row().assert_zero_ext(AB::ExprEF::from(aux_current[idx].clone().into()) - AB::ExprEF::ONE);
                    },
                    BusType::Logup => {
                        builder.when_first_row().assert_zero_ext(AB::ExprEF::from(aux_current[idx].clone().into()));
                    }
                }
                builder.when_last_row().assert_zero_ext(AB::ExprEF::from(aux_current[idx].clone().into()) - aux_bus_boundary_values[idx].into());
            }
        }
    }
}
