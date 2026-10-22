/* Deterministic inference only: median imputation, drop-first ZIP encoding,
   stored standardization and frozen LinearRegression coefficients. */
(function (root) {
  'use strict';
  function predict(model, input) {
    const p = model.predictor;
    const missing = x => x === '' || x === null || x === undefined || !Number.isFinite(Number(x));
    const x = p.numeric.map((name, i) => missing(input[name]) ? p.medians[i] : Number(input[name]));
    const zip = missing(input.zipcode) ? p.zip_fill : Number(input.zipcode);
    p.zip_categories.forEach((cat, i) => { if (i !== p.zip_drop) x.push(zip === cat ? 1 : 0); });
    return p.intercept + x.reduce((total, v, i) => total + ((v - p.mean[i]) / p.scale[i]) * p.coefficients[i], 0);
  }
  root.HW2 = {predict};
  if (typeof module !== 'undefined') module.exports = {predict};
})(globalThis);
