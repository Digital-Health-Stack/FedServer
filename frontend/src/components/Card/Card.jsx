import React from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';

const Card = ({ item, opendetails }) => {
  return (
    <div className="col">
      <div className="card h-100 border-secondary shadow-sm hover-shadow bg-light">
        <div className="card-header text-center bg-dark text-white">
          <h3>{item["org_name"]}</h3>
        </div>
        <div className="card-body">
          <h6 className="card-title">
            SessionID: {item["session_id"]}
          </h6>
          <button
            className="btn btn-dark"
            onClick={() => opendetails(item["session_id"])}
          >
            Expand
          </button>
        </div>
      </div>
    </div>
  );
};

export default Card;
