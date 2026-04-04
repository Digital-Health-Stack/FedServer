// import React, { useState, useEffect } from "react";
// import { motion, AnimatePresence } from "framer-motion";
// import { ChevronRightIcon, XMarkIcon } from "@heroicons/react/24/outline";

// const DatasetLayout = ({ children, sections }) => {
//   const [isPanelOpen, setIsPanelOpen] = useState(true);
//   const [activeSection, setActiveSection] = useState(null);

//   useEffect(() => {
//     const observer = new IntersectionObserver(
//       (entries) => {
//         entries.forEach((entry) => {
//           if (entry.isIntersecting) {
//             setActiveSection(entry.target.id);
//           }
//         });
//       },
//       { threshold: 0.5 }
//     );

//     sections.forEach((section) => {
//       const element = document.getElementById(section.id);
//       if (element) observer.observe(element);
//     });

//     return () => observer.disconnect();
//   }, [sections]);

//   return (
//     <div className="flex h-screen overflow-hidden">
//       <AnimatePresence initial={false}>
//         {isPanelOpen && (
//           <motion.div
//             initial={{ width: 0 }}
//             animate={{ width: 300 }}
//             exit={{ width: 0 }}
//             className="h-full border-r bg-gray-50 overflow-y-auto"
//           >
//             <div className="p-4 space-y-2">
//               <div className="flex justify-between items-center mb-4">
//                 <h2 className="text-lg font-semibold">Navigation</h2>
//                 <button
//                   onClick={() => setIsPanelOpen(false)}
//                   className="p-1 hover:bg-gray-200 rounded-full"
//                 >
//                   <XMarkIcon className="w-5 h-5" />
//                 </button>
//               </div>
//               {sections.map((section) => (
//                 <a
//                   key={section.id}
//                   href={`#${section.id}`}
//                   className={`block p-3 rounded-lg ${
//                     activeSection === section.id
//                       ? "bg-blue-100 text-blue-700"
//                       : "hover:bg-gray-200"
//                   }`}
//                 >
//                   <div className="flex items-center space-x-2">
//                     {section.icon}
//                     <span>{section.title}</span>
//                   </div>
//                 </a>
//               ))}
//             </div>
//           </motion.div>
//         )}
//       </AnimatePresence>

//       <div className={`flex-1 overflow-y-auto ${!isPanelOpen ? "pl-0" : ""}`}>
//         {!isPanelOpen && (
//           <button
//             onClick={() => setIsPanelOpen(true)}
//             className="fixed left-0 top-1/2 p-2 bg-white shadow-lg rounded-r-full"
//           >
//             <ChevronRightIcon className="w-5 h-5" />
//           </button>
//         )}
//         <div className="max-w-6xl mx-auto p-6">{children}</div>
//       </div>
//     </div>
//   );
// };

// export default DatasetLayout;

import React, { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ChevronRightIcon, XMarkIcon } from "@heroicons/react/24/outline";

const ICON_COLORS = [
  "text-blue-500",
  "text-green-500",
  "text-purple-500",
  "text-indigo-500",
  "text-pink-500",
];

const DatasetLayout = ({ children, sections }) => {
  const [isPanelOpen, setIsPanelOpen] = useState(true);
  const [activeSection, setActiveSection] = useState(null);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        });
      },
      { threshold: 0.5 },
    );

    sections.forEach((section) => {
      const el = document.getElementById(section.id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, [sections]);

  return (
    <div className="flex h-screen">
      <AnimatePresence initial={false}>
        {isPanelOpen && (
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: "20%" }}
            exit={{ width: 0 }}
            transition={{ type: "tween", duration: 0.3 }}
            className="h-full bg-white shadow"
          >
            <div className="p-6 sticky top-0 bg-white">
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-xl font-bold text-gray-800">Navigation</h2>
                <button
                  onClick={() => setIsPanelOpen(false)}
                  className="p-1 hover:bg-gray-100 rounded-full"
                >
                  <XMarkIcon className="w-6 h-6 text-gray-600" />
                </button>
              </div>
              <nav className="space-y-2">
                {sections.map((section, idx) => {
                  const color = ICON_COLORS[idx % ICON_COLORS.length];
                  const active = activeSection === section.id;
                  const Icon = React.cloneElement(section.icon, {
                    className: `w-6 h-6 ${color}`,
                  });
                  return (
                    <a
                      key={section.id}
                      href={`#${section.id}`}
                      className={`flex items-center p-3 rounded-lg transition duration-150 ${
                        active ? "bg-gray-100" : "hover:bg-gray-100"
                      }`}
                    >
                      {Icon}
                      <span className="ml-3 font-medium text-gray-800">
                        {section.title}
                      </span>
                    </a>
                  );
                })}
              </nav>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex-1 overflow-y-auto">
        {!isPanelOpen && (
          <button
            onClick={() => setIsPanelOpen(true)}
            className="fixed left-0 top-1/2 transform -translate-y-1/2 p-2 bg-white shadow-md rounded-r-full z-10"
          >
            <ChevronRightIcon className="w-6 h-6 text-gray-600" />
          </button>
        )}
        <div className="w-full p-6">{children}</div>
      </div>
    </div>
  );
};

export default DatasetLayout;
