/**
 * Domain shape types mirroring the taxonomy defined in ml/config/taxonomy.py.
 *
 * These are intentionally kept as plain `string` types rather than string
 * literal unions: the taxonomy is fetched at runtime from the API
 * (/api/regions, /api/categories, /api/categories/{category}/attribute-types)
 * so the frontend never hardcodes the list of valid values. The aliases
 * below exist purely for readability/self-documentation at call sites.
 */

export type Region = string;
export type Category = string;
export type AttributeType = string;
export type AttributeValue = string;
