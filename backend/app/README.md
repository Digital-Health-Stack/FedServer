# Federated Server Backend

## Setup

- Create environment with `python -m venv venv`
- Checkout environment `source venv/bin/activate`
- Install requirements `pip install -r requirements.txt`
- Create _example.env_ to _.env_ and update the values.
- Run `uvicorn main-temp:app`

## DB Setup

- Run `alembic upgrade head`

### Temporary bug fix

If any error occurs in db due to above command, go to _alembic/versions_ folder and delete all .py files and then run the above command, `alembic upgrade head`

### To create migration for new tables

- Run `alembic revision --autogenerate -m ""`
