//////////////////////////
// lccube_geometry.h
//////////////////////////
//
// Shell geometry of the cubic-lattice light cone, shared by the writer (gasgrd, lccube.hpp,
// compiled with -DLCCUBE) and the reader (snapshot-raytracer, lightcone-raytracer). Both must
// assign every grid node to exactly the same shell, bit for bit, on host and device:
//
//   - a node is identified by its unwrapped integer coordinates X_u = X - u*N (X in [0,N),
//     u the integer shift of the periodic copy v + u of the observer vertex v) plus a node
//     offset of 0 or 1/2 per axis (staggered components);
//   - displacement d = ((double) X_u + offset) / N - v, one rounding per operation;
//   - r = sqrt(fma(d2, d2, fma(d1, d1, d0*d0))), cone: fma(d2, n2, fma(d1, n1, d0*n0)) / r;
//     every operation is explicit (device intrinsics, std::fma on the host), so no compiler
//     contraction or reassociation can change a result;
//   - shells are half-open [inner, outer); boundaries come from the binary shell log.
//
// The geometry file pair <basename>_lccube_geometry.bin / .txt records the parameters as the
// writer used them (binary: exact; text: for reading by humans).
//
// No dependencies on either code base. Do not compile with -ffast-math.
//
//////////////////////////

#ifndef LCCUBE_GEOMETRY_H
#define LCCUBE_GEOMETRY_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#if defined(__FAST_MATH__)
#error "lccube_geometry.h requires IEEE double arithmetic: do not compile with -ffast-math"
#endif

#if defined(__CUDACC__)
#define LCG_HD __host__ __device__
#else
#define LCG_HD
#endif

namespace lcg {

constexpr std::int32_t CONVENTION = 1; // version of the arithmetic below (stored in the geometry file)
constexpr int MAX_COPIES = 343;        // periodic copies per shell and process (7^3)

// IEEE round-to-nearest operations that the compiler may not fuse or reorder
LCG_HD inline double add(double a, double b)
{
#ifdef __CUDA_ARCH__
	return __dadd_rn(a, b);
#else
	return a + b;
#endif
}
LCG_HD inline double sub(double a, double b)
{
#ifdef __CUDA_ARCH__
	return __dsub_rn(a, b);
#else
	return a - b;
#endif
}
LCG_HD inline double mul(double a, double b)
{
#ifdef __CUDA_ARCH__
	return __dmul_rn(a, b);
#else
	volatile double p = a * b; // keeps the product a separate rounding even with -ffp-contract=fast
	return p;
#endif
}
LCG_HD inline double div(double a, double b)
{
#ifdef __CUDA_ARCH__
	return __ddiv_rn(a, b);
#else
	return a / b;
#endif
}
LCG_HD inline double fma(double a, double b, double c)
{
#ifdef __CUDA_ARCH__
	return __fma_rn(a, b, c);
#else
	return std::fma(a, b, c);
#endif
}
LCG_HD inline double sqrt(double a)
{
#ifdef __CUDA_ARCH__
	return __dsqrt_rn(a);
#else
	return std::sqrt(a);
#endif
}

// displacement (box units) of the node with unwrapped coordinates xyz and offset from vertex v
LCG_HD inline void displacement(const std::int64_t xyz[3], const double offset[3], const double n, const double v[3], double d[3])
{
	for (int a = 0; a < 3; a++)
		d[a] = sub(div(add((double) xyz[a], offset[a]), n), v[a]);
}

LCG_HD inline double radius(const double d[3])
{
	return lcg::sqrt(lcg::fma(d[2], d[2], lcg::fma(d[1], d[1], mul(d[0], d[0]))));
}

// opening_cos <= -1: full sky; otherwise the node must lie strictly inside the cone
LCG_HD inline bool in_cone(const double d[3], const double r, const double direction[3], const double opening_cos)
{
	if (!(opening_cos > -1.)) return true;
	if (!(r > 0.)) return false;
	return div(lcg::fma(d[2], direction[2], lcg::fma(d[1], direction[1], mul(d[0], direction[0]))), r) > opening_cos;
}

LCG_HD inline bool in_shell(const double r, const double inner, const double outer)
{
	return r >= inner && r < outer;
}

// radius of a node and whether it lies in the cone (r < 0 if the radius is not finite)
LCG_HD inline double node_radius(const std::int64_t xyz[3], const double offset[3], const double n, const double v[3], const double direction[3], const double opening_cos, bool & cone)
{
	double d[3];
	displacement(xyz, offset, n, v, d);
	const double r = radius(d);
	if (!(r == r) || r > 1.7976931348623157e308) { cone = false; return -1.; }
	cone = in_cone(d, r, direction, opening_cos);
	return r;
}

// whether the writer fills slot `slot` of a record with shell radii s[0..T] (distance range in box units)
LCG_HD inline bool slot_written(const double * s, const int slot, const double distance_outer, const double distance_inner)
{
	return !(distance_outer <= s[slot+1] || distance_inner > s[slot+1] || s[slot+1] <= 0.);
}

// whether the writer logs a record at all (otherwise it neither fills nor logs anything)
inline bool record_written(const double * s, const int thickness, const double distance_outer, const double distance_inner)
{
	return distance_outer > s[1] && distance_inner <= s[thickness] && s[thickness] > 0.;
}

// periodic copies u (shift of the vertex) whose shell [inner, outer) can reach nodes in the box
// [lo, hi] (box units, already padded by the caller); conservative: no angular culling. Returns the
// number of copies found; only the first `capacity` are stored.
inline int enumerate_copies(const double v[3], const double outer, const double inner, const double lo[3], const double hi[3], std::int64_t (*out)[3], const int capacity)
{
	const double margin = 1e-9;
	const int range = (int) std::ceil(outer) + 1;
	int count = 0;
	for (int u = -range; u <= range; u++)
	for (int w = -range; w <= range; w++)
	for (int t = -range; t <= range; t++)
	{
		const int shift[3] = {u, w, t};
		double near2 = 0., far2 = 0.;
		for (int a = 0; a < 3; a++)
		{
			const double c = v[a] + shift[a];
			const double q = c < lo[a] ? lo[a] : (c > hi[a] ? hi[a] : c);
			const double f = std::fabs(c - lo[a]) > std::fabs(c - hi[a]) ? std::fabs(c - lo[a]) : std::fabs(c - hi[a]);
			near2 += (c - q) * (c - q);
			far2 += f * f;
		}
		if (std::sqrt(near2) > outer + margin || std::sqrt(far2) < inner - margin) continue;
		if (count < capacity)
			for (int a = 0; a < 3; a++) out[count][a] = shift[a];
		count++;
	}
	return count;
}

//////////////////////////
// geometry file
//////////////////////////

enum : std::int32_t { FIELD_PHI = 1, FIELD_CHI = 2, FIELD_B = 4, FIELD_HIJ = 8, FIELD_WEYL = 16, FIELD_RECORD = 32 };

struct GeometryInfo
{
	std::int32_t convention = CONVENTION;
	std::int32_t n = 0;                 // grid points per dimension
	std::int32_t thickness = 0;         // slots per field
	std::int32_t b_collocated = 0;      // 1: B interpolated to the lattice sites (LIGHTCONE_INTERPOLATE)
	std::int32_t downgrade = 1;         // hij lattice coarsening factor
	std::int32_t fields = 0;            // FIELD_* bits
	std::int32_t periodic_overlap = 0;  // 1: footprint may overlap its periodic images (distance_outer > 1/2)
	std::int32_t lightcone = 0;         // light cone index in the settings file
	double vertex[3] = {0., 0., 0.};    // box units
	double direction[3] = {0., 0., 1.}; // normalised
	double opening_cos = -1.;           // -1: full sky
	double distance_outer = 0.;         // box units
	double distance_inner = 0.;
	double tauobs = 0.;                 // tau/boxsize of the observer
	double boxsize = 0.;                // Mpc/h
};

constexpr char GEOMETRY_MAGIC[8] = {'L', 'C', 'C', 'U', 'B', 'E', 'G', 'M'};
constexpr std::int32_t GEOMETRY_INTS = 8;
constexpr std::int32_t GEOMETRY_DOUBLES = 11;
constexpr long GEOMETRY_BYTES = 8 + 4 + 4 * GEOMETRY_INTS + 8 * GEOMETRY_DOUBLES;

// binary, native byte order: magic, int32 count of int32 fields, int32 fields, float64 fields
inline bool write_geometry_binary(const std::string & path, const GeometryInfo & g)
{
	FILE * f = std::fopen(path.c_str(), "wb");
	if (f == NULL) return false;
	const std::int32_t ints[GEOMETRY_INTS] = {g.convention, g.n, g.thickness, g.b_collocated, g.downgrade, g.fields, g.periodic_overlap, g.lightcone};
	const double doubles[GEOMETRY_DOUBLES] = {g.vertex[0], g.vertex[1], g.vertex[2], g.direction[0], g.direction[1], g.direction[2],
		g.opening_cos, g.distance_outer, g.distance_inner, g.tauobs, g.boxsize};
	bool ok = std::fwrite(GEOMETRY_MAGIC, 1, 8, f) == 8;
	ok = ok && std::fwrite(&GEOMETRY_INTS, sizeof(std::int32_t), 1, f) == 1;
	ok = ok && std::fwrite(ints, sizeof(std::int32_t), GEOMETRY_INTS, f) == (size_t) GEOMETRY_INTS;
	ok = ok && std::fwrite(doubles, sizeof(double), GEOMETRY_DOUBLES, f) == (size_t) GEOMETRY_DOUBLES;
	return std::fclose(f) == 0 && ok;
}

inline GeometryInfo read_geometry_binary(const std::string & path)
{
	FILE * f = std::fopen(path.c_str(), "rb");
	if (f == NULL) throw std::runtime_error("Cannot open light cone geometry file: " + path);
	char magic[8];
	std::int32_t count = 0, ints[GEOMETRY_INTS];
	double doubles[GEOMETRY_DOUBLES];
	bool ok = std::fread(magic, 1, 8, f) == 8 && std::memcmp(magic, GEOMETRY_MAGIC, 8) == 0;
	ok = ok && std::fread(&count, sizeof(std::int32_t), 1, f) == 1 && count == GEOMETRY_INTS;
	ok = ok && std::fread(ints, sizeof(std::int32_t), GEOMETRY_INTS, f) == (size_t) GEOMETRY_INTS;
	ok = ok && std::fread(doubles, sizeof(double), GEOMETRY_DOUBLES, f) == (size_t) GEOMETRY_DOUBLES;
	ok = ok && std::fgetc(f) == EOF;
	std::fclose(f);
	if (!ok) throw std::runtime_error("Not a light cone geometry file (or wrong byte order/version): " + path);
	GeometryInfo g;
	g.convention = ints[0]; g.n = ints[1]; g.thickness = ints[2]; g.b_collocated = ints[3];
	g.downgrade = ints[4]; g.fields = ints[5]; g.periodic_overlap = ints[6]; g.lightcone = ints[7];
	for (int a = 0; a < 3; a++) { g.vertex[a] = doubles[a]; g.direction[a] = doubles[3+a]; }
	g.opening_cos = doubles[6]; g.distance_outer = doubles[7]; g.distance_inner = doubles[8];
	g.tauobs = doubles[9]; g.boxsize = doubles[10];
	if (g.convention != CONVENTION)
		throw std::runtime_error("Light cone geometry convention " + std::to_string(g.convention) + " is not supported (expected " + std::to_string(CONVENTION) + "): " + path);
	if (g.n <= 0 || g.thickness <= 0 || g.downgrade <= 0)
		throw std::runtime_error("Invalid light cone geometry file: " + path);
	return g;
}

// human-readable copy (round-trip precision); the binary file is authoritative
inline bool write_geometry_text(const std::string & path, const GeometryInfo & g)
{
	FILE * f = std::fopen(path.c_str(), "w");
	if (f == NULL) return false;
	std::fprintf(f, "# light cone cube geometry (lightcone %d); lengths and times in units of the box size\n", g.lightcone);
	std::fprintf(f, "# the binary file of the same name is authoritative; values below are %%.17g\n");
	std::fprintf(f, "convention       = %d\n", g.convention);
	std::fprintf(f, "grid             = %d\n", g.n);
	std::fprintf(f, "thickness        = %d\n", g.thickness);
	std::fprintf(f, "B layout         = %s\n", g.b_collocated ? "collocated" : "staggered");
	std::fprintf(f, "hij downgrade    = %d\n", g.downgrade);
	std::fprintf(f, "fields           =%s%s%s%s%s%s\n", (g.fields & FIELD_PHI) ? " phi" : "", (g.fields & FIELD_CHI) ? " chi" : "", (g.fields & FIELD_B) ? " B" : "",
		(g.fields & FIELD_HIJ) ? " hij" : "", (g.fields & FIELD_WEYL) ? " weyl" : "", (g.fields & FIELD_RECORD) ? " recordcycle" : "");
	std::fprintf(f, "periodic overlap = %s\n", g.periodic_overlap ? "possible" : "no");
	std::fprintf(f, "vertex           = %.17g %.17g %.17g\n", g.vertex[0], g.vertex[1], g.vertex[2]);
	std::fprintf(f, "direction        = %.17g %.17g %.17g\n", g.direction[0], g.direction[1], g.direction[2]);
	std::fprintf(f, "opening cosine   = %.17g\n", g.opening_cos);
	std::fprintf(f, "distance (outer, inner) = %.17g %.17g\n", g.distance_outer, g.distance_inner);
	std::fprintf(f, "tauobs           = %.17g\n", g.tauobs);
	std::fprintf(f, "boxsize [Mpc/h]  = %.17g\n", g.boxsize);
	return std::fclose(f) == 0;
}

} // namespace lcg

#endif
