import React, { useState } from "react";

interface Column<T> {
  label: string;
  key?: string;
  render: (item: T, index: number) => React.ReactNode;
}

interface DataTableProps<T> {
  tableStyle?:string,
  data: T[];
  columns: Column<T>[];
  minRows?: number;
  getStatusStyle?: (value: string) => string;
  pagination?: boolean;
}

const DataTable = <T extends unknown>({
  tableStyle,
  data,
  columns,
  minRows = 10,
  getStatusStyle,
  pagination = true,
}: DataTableProps<T>) => {
  const [currentPage, setCurrentPage] = useState(1);

  const totalPages = Math.ceil(data.length / minRows);
  const startIndex = pagination ? (currentPage - 1) * minRows : 0;
  const pageData = pagination
    ? data.slice(startIndex, startIndex + minRows)
    : data;

  const filledRows = [...pageData];
  while (pagination && filledRows.length < minRows) {
    filledRows.push(null as unknown as T);
  }

  const handlePrev = () => pagination && currentPage > 1 && setCurrentPage(currentPage - 1);
  const handleNext = () => pagination && currentPage < totalPages && setCurrentPage(currentPage + 1);

  const columnClassMap: { [key: number]: string } = {
    1: "grid-cols-1",
    2: "grid-cols-2",
    3: "grid-cols-3",
    4: "grid-cols-4",
    5: "grid-cols-5",
    6: "grid-cols-6",
    7: "grid-cols-7",
    8: "grid-cols-8",
    9: "grid-cols-9",
    10: "grid-cols-10",
  };

  const gridCol = columnClassMap[columns.length] || "grid-cols-1";

  return (
    <div className="w-full">
      <div className={`grid ${gridCol} ${tableStyle} gap-3 bg-primary text-white font-title font-medium text rounded-xl px-6 py-3`}>
        {columns.map((col, idx) => (
          <div key={idx}>{col.label}</div>
        ))}
      </div>
      <div className="h-4" />
      <div className="rounded-xl overflow-hidden">
        {filledRows.map((item, index) => (
          <div
            key={index}
            className={`grid ${gridCol} gap-5 border-b border-color last:border-0 font-medium foreground paragraph font-paragraph  px-6 py-4 shadow-sm text-xs hover:shadow-md transition-all duration-300`}
          >
            {item
              ? columns.map((col, colIndex) => {
                  const value = col.key && (item as any)[col.key];
                  const isStatus = col.key === "status";

                  return (
                    <div key={colIndex}>
                      {isStatus && getStatusStyle ? (
                        <span
                          className={`px-4 py-2 rounded-md text-xs font-medium capitalize ${getStatusStyle(
                            value
                          )}`}
                        >
                          {value}
                        </span>
                      ) : (
                        col.render(item, startIndex + index)
                      )}
                    </div>
                  );
                })
              : columns.map((_, colIndex) => (
                  <div key={colIndex} className="text-gray-300">
                    -
                  </div>
                ))}
          </div>
        ))}
      </div>

      {/* Pagination Controls */}
      {pagination && totalPages > 0 && (
        <div className="flex justify-between items-center mt-4 px-3 py-3 border border-color rounded-xl foreground text-sm ">
          <button
            onClick={handlePrev}
            disabled={currentPage === 1}
            className={`px-5 py-2 rounded-lg font-medium transition-all duration-200 ${
              currentPage === 1
                ? "bg-gray-200 dark:bg-gray-700 text-gray-400 cursor-not-allowed"
                : "bg-primary text-white hover:opacity-90"
            }`}
            aria-label="Previous Page"
          >
          Previous
          </button>

          <div className="text paragraph font-title select-none">
            Page{" "}
            <span className="font-semibold text-text-light dark:text-text-dark">{currentPage}</span>{" "}
            of{" "}
            <span className="font-semibold text-text-light dark:text-text-dark">{totalPages}</span>
          </div>

          <button
            onClick={handleNext}
            disabled={currentPage === totalPages}
            className={`px-5 py-2 rounded-lg font-medium transition-all duration-200 ${
              currentPage === totalPages
                ? "bg-gray-200 dark:bg-gray-700 text-gray-400 cursor-not-allowed"
                : "bg-primary text-white hover:opacity-90"
            }`}
            aria-label="Next Page"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
};

export default DataTable;
