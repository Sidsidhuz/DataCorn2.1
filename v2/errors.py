"""User-facing failures have stable codes; internal exceptions stay in logs."""
class ProductError(ValueError):
    def __init__(self, message, code='invalid_request'):
        super().__init__(message)
        self.code = code


class NotFound(ProductError):
    def __init__(self):
        super().__init__('This item is unavailable in your workspace.', 'not_found')


class JobCancelled(ProductError):
    def __init__(self):
        super().__init__('Training was cancelled. No partial model was published.', 'cancelled')
