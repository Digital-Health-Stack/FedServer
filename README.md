# Federated Server Backend

# To Run the application

1. Create .env for both frontend and backend
2. Setup DB based on below DB instruction
3. Download data for backend/app/utility/global_test_data
4. docker-compose up --build

## DB Setup

- Run `alembic upgrade head`

### Temporary bug fix

If any error occurs in db due to above command, go to _alembic/versions_ folder and delete all .py files and then run the above command, `alembic upgrade head`

### To create migration for new tables

- Run `alembic revision --autogenerate -m ""`
- Run `alembic upgrade head`

some useful cmds-
`uvicorn main:app --lazy-imports`
`uvicorn main:app --reload --reload-dir=app/`

<!-- read spark logs -->

```
 tail -n 10 -f logs.out
```
