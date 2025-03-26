import { CubeIcon, DocumentTextIcon } from "@heroicons/react/24/solid";

export default function PreprocessingDocs() {
  return (
    <div className="max-w-5xl mx-auto p-8 bg-white border border-gray-200 rounded-lg">
      {/* Header */}
      <header className="mb-8 text-center">
        <h1 className="text-3xl font-bold text-gray-800 flex items-center justify-center gap-3 mb-2">
          <CubeIcon className="h-8 w-8" />
          Data Preprocessing Guide
        </h1>
        <p className="text-gray-600">
          A simple guide explaining available operations, recommended order, and
          precautions.
        </p>
      </header>

      {/* General Workflow & Order */}
      <section className="mb-6">
        <h2 className="text-2xl font-semibold text-gray-700 mb-2">
          General Workflow & Order
        </h2>
        <p className="text-gray-600 mb-4">
          To obtain the best results, operations should be applied in a proper
          sequence:
        </p>
        <ol className="list-decimal list-inside text-gray-600">
          <li>
            <strong>Data Cleaning:</strong> Begin with handling nulls (either
            drop or fill them) and drop duplicate rows. These steps ensure that
            later operations (such as scaling or encoding) are not disrupted by
            unexpected values.
          </li>
          <li>
            <strong>Outlier Removal:</strong> Apply outlier treatment (using the
            IQR method) next. Note that if any numeric column has an outlier,
            the entire row will be removed.
          </li>
          <li>
            <strong>Normalization/Scaling:</strong> Next, use normalization or
            scaling methods on numeric columns. When applied in the “All
            Columns” mode, the operation treats each row as a vector – so be
            sure to exclude any nonnumeric columns.
          </li>
          <li>
            <strong>Transformations:</strong> Operations such as logarithm,
            square, or square root should be applied only after cleaning,
            outlier removal, and normalization. These functions expect positive
            or nonzero inputs.
          </li>
          <li>
            <strong>Encoding:</strong> For string or categorical columns, apply
            encoding (Label Encoding or One-Hot Encoding) once data cleaning
            steps are complete.
          </li>
        </ol>
      </section>

      {/* Operations on All Columns */}
      <section className="mb-6">
        <h2 className="text-2xl font-semibold text-gray-700 mb-2">
          All Columns Operations
        </h2>
        <p className="text-gray-600 mb-4">
          These operations are applied across the entire dataset:
        </p>
        <ul className="list-disc list-inside text-gray-600">
          <li>
            <strong>Handle Null:</strong> Options include dropping all rows with
            any null value or filling nulls with defaults (0 for numeric,
            'unknown' for strings, or false for booleans).
          </li>
          <li>
            <strong>Drop Duplicates:</strong> Removes duplicate rows across all
            selected columns.
          </li>
          <li>
            <strong>Normalization:</strong> When applied in “All Columns” mode,
            a row’s numeric column (except columns excluded from All columns
            list) values are treated as a vector and normalized (this internally
            ignores non-numeric columns), however final result is not a vector
            instead previous values are replaced by new normalized value for
            each column.
          </li>
          <li>
            <strong>Remove Outliers:</strong> Uses the IQR method to detect and
            remove rows with outliers in any numeric column.
          </li>
        </ul>
      </section>

      {/* Numeric Column Specific Operations */}
      <section className="mb-6">
        <h2 className="text-2xl font-semibold text-gray-700 mb-2">
          Numeric Column Specific Operations
        </h2>
        <p className="text-gray-600 mb-4">
          When dealing with numeric data, additional operations are available:
        </p>
        <ul className="list-disc list-inside text-gray-600">
          <li>
            <strong>Drop Column:</strong> Remove a numeric column entirely.
          </li>
          <li>
            <strong>Handle Null:</strong> Fill null values with 0, mean, median,
            or mode—choose the method that best suits your data distribution.
          </li>
          <li>
            <strong>Normalization/Scaling:</strong> Supported methods include
            Min-Max, Z-score, L1 Norm, L2 Norm, and L-inf Norm. These methods
            compute scaling based on the data’s statistics; if a column is
            constant, it will be replaced with 0.0.
          </li>
          <li>
            <strong>Transformations:</strong> Mathematical functions (Log,
            Square, Square Root) are available, but verify that values are valid
            (e.g., positive for log and square root) to avoid errors.
          </li>
          <li>
            <strong>Remove Outliers:</strong> Uses the IQR method to remove rows
            containing outlier values.
          </li>
        </ul>
      </section>

      {/* String Column Specific Operations */}
      <section className="mb-6">
        <h2 className="text-2xl font-semibold text-gray-700 mb-2">
          String Column Specific Operations
        </h2>
        <p className="text-gray-600 mb-4">
          For columns that contain textual or categorical data, consider these
          operations:
        </p>
        <ul className="list-disc list-inside text-gray-600">
          <li>
            <strong>Drop Column:</strong> Remove the column if it is not needed.
          </li>
          <li>
            <strong>Handle Null:</strong> Either drop rows that have null values
            or fill them with a default value (for example, "Unknown").
          </li>
          <li>
            <strong>Drop Duplicates:</strong> Remove duplicate entries within
            the column.
          </li>
          <li>
            <strong>Encode:</strong> Convert categorical data to a numerical
            format using Label Encoding (for ordinal data) or One-Hot Encoding
            (for nominal data). Note that certain encoding operations work only
            on integer-compatible data.
          </li>
        </ul>
      </section>

      {/* Precautions & Recommendations */}
      <section>
        <h2 className="text-2xl font-semibold text-gray-700 mb-2">
          Precautions & Recommendations
        </h2>
        <p className="text-gray-600 mb-4">
          Few points to note to help avoid potential issues:
        </p>
        <ul className="list-disc list-inside text-gray-600">
          <li>
            <strong>Start with cleaning:</strong> Always handle nulls and remove
            duplicates first. Applying normalization or encoding before these
            steps may produce errors or unexpected results.
          </li>
          <li>
            <strong>Order matters:</strong> Wrong sequencing (for example,
            transforming data before outlier removal) can lead to loss of
            valuable data or incorrect scaling.
          </li>
          <li>
            <strong>Data validity:</strong> Check that inputs are valid when
            applying mathematical transformations (e.g., logarithms are defined
            only for positive numbers, Encode data only when no NULL values are
            there.)
          </li>
          <li>
            <strong>Exclude from All Columns:</strong> This options removes that
            column from all the All Columns operations afterwards. (useful for
            seperating output columns from the bulk operations in input
            columns).
          </li>
          <li>
            <strong>Order of Columns:</strong> After all the operations order of
            the columns are mantained as before in the dataset.
          </li>
        </ul>
      </section>

      <footer className="mt-8 text-gray-600">
        <DocumentTextIcon className="inline h-6 w-6 mr-2" />
        <span>
          This guide is intended as a simple reference for selecting and
          ordering preprocessing operations. Adjust the workflow as needed based
          on the specific characteristics of your dataset.
        </span>
      </footer>
    </div>
  );
}
