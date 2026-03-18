class SalesRecord:
    '''Represents a single sales transaction.'''
    def __init__(self, store_id, product, amount, date='2025-01-01'):
        self.store_id = store_id
        self.product = product
        self.amount = amount
        self.date = date

    def summary(self):
        print(f'Store {self.store_id} | {self.product} | ${self.amount:,.2f} | {self.date}')


class SalesReport:
    '''Collects and analyzes a set of SalesRecord objects.'''
    def __init__(self, region):
        self.region = region
        self.records = []          # default: empty list, grows as records are added

    def add_record(self, record):
        self.records.append(record)

    def total_sales(self):
        return sum(r.amount for r in self.records)

    def top_sale(self):
        return max(self.records, key=lambda r: r.amount)

    def print_report(self):
        print(f'\n=== {self.region} Region Report ===')
        print(f'Transactions: {len(self.records)}')
        print(f'Total Sales:  ${self.total_sales():,.2f}')
        top = self.top_sale()
        print(f'Top Sale:     {top.product} at ${top.amount:,.2f}')


  # Create some records
r1 = SalesRecord(101, 'Hiking Boots', 129.99, '2025-03-01')
r2 = SalesRecord(101, 'Tent', 349.99, '2025-03-02')
r3 = SalesRecord(204, 'Water Bottle', 24.99, '2025-03-03')

r1.summary()    # calling a method on a SalesRecord instance

  # Build a report by adding records to it
east = SalesReport('East')
east.add_record(r1)
east.add_record(r2)
east.add_record(r3)
east.print_report()