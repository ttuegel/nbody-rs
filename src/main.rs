extern crate simd;

use std::marker;
use simd::x86::sse2::*;
use simd::x86::sse3::*;

const PI : f64 = 3.141592653589793;
const SOLAR_MASS : f64 = 4.0 * PI * PI;
const DAYS_PER_YEAR : f64 = 365.24;
const TIME_RESOLUTION : f64 = 0.01;

struct Pairs<'a, T> where T : 'a {
    ptr_a : *const T,
    ptr_b : *const T,
    end : *const T,
    _marker : marker::PhantomData<&'a [T]>,
}

fn pairs<'a, T>(slice : &'a [T]) -> Pairs<'a, T> {
    let ptr_a = slice.as_ptr();
    let ptr_b = unsafe { ptr_a.offset(1) };
    let end = unsafe { ptr_a.offset(slice.len() as isize) };
    Pairs {
        ptr_a : ptr_a,
        ptr_b : ptr_b,
        end : end,
        _marker : marker::PhantomData,
    }
}

impl<'a, T> Iterator for Pairs<'a, T> {
    type Item = (&'a T, &'a T);
    fn next(&mut self) -> Option<(&'a T, &'a T)> {
        let ptr_a = self.ptr_a;
        let ptr_b = self.ptr_b;
        if ptr_b < self.end {
            unsafe {
                let a = ptr_a.as_ref().unwrap();
                let b = ptr_b.as_ref().unwrap();
                self.ptr_b = ptr_b.offset(1);
                Some((a, b))
            }
        } else {
            let ptr_a = unsafe { ptr_a.offset(1) };
            let ptr_b = unsafe { ptr_a.offset(1) };
            if ptr_b < self.end {
                unsafe {
                    let a = ptr_a.as_ref().unwrap();
                    let b = ptr_b.as_ref().unwrap();
                    self.ptr_a = ptr_a;
                    self.ptr_b = ptr_b.offset(1);
                    Some((a, b))
                }
            } else {
                None
            }
        }
    }
}

struct PairsMut<'a, T> where T : 'a {
    ptr_a : *mut T,
    ptr_b : *mut T,
    end : *mut T,
    _marker : marker::PhantomData<&'a mut [T]>,
}

fn pairs_mut<'a, T>(slice : &'a mut [T]) -> PairsMut<'a, T> {
    let ptr_a : *mut T = slice.as_mut_ptr();
    let ptr_b : *mut T = unsafe { ptr_a.offset(1) };
    let end : *mut T = unsafe { ptr_a.offset(slice.len() as isize) };
    PairsMut {
        ptr_a : ptr_a,
        ptr_b : ptr_b,
        end : end,
        _marker : marker::PhantomData,
    }
}

impl<'a, T> Iterator for PairsMut<'a, T> {
    type Item = (&'a mut T, &'a mut T);
    fn next(&mut self) -> Option<(&'a mut T, &'a mut T)> {
        let ptr_a = self.ptr_a;
        let ptr_b = self.ptr_b;
        let end = self.end;
        if ptr_b < end {
            unsafe {
                let a = ptr_a.as_mut().unwrap();
                let b = ptr_b.as_mut().unwrap();
                self.ptr_b = ptr_b.offset(1);
                Some((a, b))
            }
        } else {
            let ptr_a = unsafe { ptr_a.offset(1) };
            let ptr_b = unsafe { ptr_a.offset(1) };
            if ptr_b < end {
                unsafe {
                    let a = ptr_a.as_mut().unwrap();
                    let b = ptr_b.as_mut().unwrap();
                    self.ptr_a = ptr_a;
                    self.ptr_b = ptr_b.offset(1);
                    Some((a, b))
                }
            } else {
                None
            }
        }
    }
}

#[derive(Default)]
struct Body {
    x : [f64; 4],
    v : [f64; 4],
    mass : f64,
}

fn mom(body : &Body) -> Vec<f64> {
    let mass = body.mass;
    body.v.iter().map(|vi| { vi * mass }).collect()
}

fn sol(bodies : &[Body]) -> Body {
    Body {
        x : [0.0; 4],
        v : {
            let mut v : [f64; 4] = [0.0; 4];
            for (i, v_i) in v.iter_mut().enumerate() {
                *v_i = - bodies.iter().map(|bd| { mom(bd)[i] }).sum::<f64>()
                    / SOLAR_MASS;
            };
            v
        },
        mass : SOLAR_MASS,
    }
}

fn kinetic_energy(bd : &Body) -> f64 {
    bd.v.iter().map(|vi| { vi.powi(2) }).sum::<f64>() * bd.mass / 2.0
}

fn potential_energy(pair : (&Body, &Body)) -> f64 {
    let (a, b) = pair;
    let dx : Vec<f64> = {
        let over_x = a.x.iter().zip(b.x.iter());
        over_x.map(|(a, b)| { a - b }).collect()
    };
    let r = dx.iter().map(|xi| { xi.powi(2) }).sum::<f64>().sqrt();
    (- a.mass) * b.mass / r
}

fn energy(bodies : &[Body]) -> f64 {
    let ke = bodies.iter().map(kinetic_energy).sum::<f64>();
    let pe = pairs(bodies).map(potential_energy).sum::<f64>();
    ke + pe
}

fn move_bodies(bds : &mut [Body]) {
    for b in bds {
        for (x_i, v_i) in b.x.iter_mut().zip(b.v.iter()) {
            *x_i += TIME_RESOLUTION * v_i;
        }
    }
}

const N : usize = 5;
const NPAIRS : usize = N * (N - 1) / 2;
const NPAD : usize = NPAIRS + 1;

fn accelerate_bodies(bodies : &mut [Body]) {
    let mut dxs : [[f64; 4]; NPAD] = [[0.0; 4]; NPAD];
    let mut fs : [f64; NPAD] = [0.0; NPAD];

    for (i, (a, b)) in pairs(bodies).enumerate() {
        {
            let a_x = f64x2::load(&a.x, 0);
            let b_x = f64x2::load(&b.x, 0);
            (a_x - b_x).store(&mut dxs[i], 0);
        }
        {
            let a_x = f64x2::load(&a.x, 2);
            let b_x = f64x2::load(&b.x, 2);
            (a_x - b_x).store(&mut dxs[i], 2);
        }
    }

    for i in 0..NPAD {
        let mut xx : [f64; 4] = [0.0; 4];
        {
            let lo = f64x2::load(&dxs[i], 0);
            (lo * lo).store(&mut xx, 0);
        }
        {
            let hi = f64x2::load(&dxs[i], 2);
            (hi * hi).store(&mut xx, 2);
        }
        fs[i] = xx.iter().sum::<f64>();
    }

    let dt = f64x2::splat(TIME_RESOLUTION);
    for i in 0..(NPAD / 2) {
        let rr = f64x2::load(&fs, 2 * i);
        let r = rr.sqrt();
        (dt / (rr * r)).store(&mut fs, 2 * i);
    }

    for (i, (a, b)) in pairs_mut(bodies).enumerate() {
        let forces = {
            let masses = f64x2::new(b.mass, a.mass);
            let f = f64x2::splat(fs[i]);
            masses * f
        };

        for j in 0..3usize {
            let v1 = {
                let v0 = f64x2::new(a.v[j], b.v[j]);
                let dv = f64x2::splat(dxs[i][j]) * forces;
                v0.addsub(dv)
            };
            a.v[j] = v1.extract(0);
            b.v[j] = v1.extract(1);
        }
    }
}

fn main() {

    let jupiter : Body = Body {
        x : [
            4.84143144246472090e+00,
            -1.16032004402742839e+00,
            -1.03622044471123109e-01,
            0.0
        ],
        v : [
            1.66007664274403694e-03 * DAYS_PER_YEAR,
            7.69901118419740425e-03 * DAYS_PER_YEAR,
            -6.90460016972063023e-05 * DAYS_PER_YEAR,
            0.0
        ],
        mass : 9.54791938424326609e-04 * SOLAR_MASS,
    };

    let saturn : Body = Body {
        x : [
            8.34336671824457987e+00,
            4.12479856412430479e+00,
            -4.03523417114321381e-01,
            0.0
        ],
        v : [
            -2.76742510726862411e-03 * DAYS_PER_YEAR,
            4.99852801234917238e-03 * DAYS_PER_YEAR,
            2.30417297573763929e-05 * DAYS_PER_YEAR,
            0.0
        ],
        mass : 2.85885980666130812e-04 * SOLAR_MASS,
    };

    let uranus : Body = Body {
        x : [
            1.28943695621391310e+01,
            -1.51111514016986312e+01,
            -2.23307578892655734e-01,
            0.0
        ],
        v : [
            2.96460137564761618e-03 * DAYS_PER_YEAR,
            2.37847173959480950e-03 * DAYS_PER_YEAR,
            -2.96589568540237556e-05 * DAYS_PER_YEAR,
            0.0
        ],
        mass : 4.36624404335156298e-05 * SOLAR_MASS,
    };

    let neptune : Body = Body {
        x : [
            1.53796971148509165e+01,
            -2.59193146099879641e+01,
            1.79258772950371181e-01,
            0.0
        ],
        v : [
            2.68067772490389322e-03 * DAYS_PER_YEAR,
            1.62824170038242295e-03 * DAYS_PER_YEAR,
            -9.51592254519715870e-05 * DAYS_PER_YEAR,
            0.0
        ],
        mass : 5.15138902046611451e-05 * SOLAR_MASS,
    };

    let mut bodies : [Body; N] = [
        Default::default(),
        jupiter,
        saturn,
        uranus,
        neptune
    ];
    bodies[0] = sol(&bodies[1..]);

    match std::env::args().nth(1) {
        Some(arg) => {
            let n = arg.parse::<isize>().unwrap();

            println!("{:.9}", energy(&bodies[..]));

            for _ in 0..n {
                accelerate_bodies(&mut bodies);
                move_bodies(&mut bodies);
            }

            println!("{:.9}", energy(&bodies[..]));
        },
        None => (),
    }

}
