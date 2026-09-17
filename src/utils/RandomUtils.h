#pragma once

#include <cstdint>

/*!
 *	Draws a fresh uint32_t seed from a properly-seeded std::mt19937, spanning the full uint32_t range.
 *	@return		A random seed suitable for TerrainParams::seed.
 */
uint32_t generateRandomSeed();

/*!
 *	Draws a random Hurst exponent, inset from the slider's full [0,1] range to avoid the
 *	visually-degenerate extremes (near-flat at 1, near-white-noise at 0).
 *	@return		A random Hurst exponent in [0.4, 0.95].
 */
float generateRandomHurst();

/*!
 *	Draws a random terrain height scale.
 *	@return		A random height scale in [1.0, 5.0].
 */
float generateRandomHeightScale();

/*!
 *	Draws a random water level, inset from the slider's full [-25, 25] range — comfortably varied
 *	without drifting the water plane absurdly far from where the terrain actually sits.
 *	@return		A random water level in [-15.0, 15.0].
 */
float generateRandomWaterLevel();
