from sqlalchemy.orm import Session


class ClassificationRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_or_create(self, model, name: str):
        instance = self.db.query(model).filter_by(name=name).first()
        if not instance:
            instance = model(name=name)
            self.db.add(instance)
            self.db.flush()
        return instance
