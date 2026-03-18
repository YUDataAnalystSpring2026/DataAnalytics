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

    def average_sale(self):        # new method added in this example to add a sales average
        if not self.records:
            return 0
        return self.total_sales() / len(self.records)

    def top_sale(self):
        return max(self.records, key=lambda r: r.amount)

    def print_report(self):
        print(f'\n=== {self.region} Region Report ===')
        print(f'Transactions: {len(self.records)}')
        print(f'Total Sales:  ${self.total_sales():,.2f}')
        print(f'Average Sale: ${self.average_sale():,.2f}')
        top = self.top_sale()
        print(f'Top Sale:     {top.product} at ${top.amount:,.2f}')


  # Create some records
r1 = SalesRecord(101, 'Hiking Boots', 129.99, '2025-03-01')
r2 = SalesRecord(101, 'Tent', 349.99, '2025-03-02')
r3 = SalesRecord(204, 'Water Bottle', 24.99, '2025-03-03')


class StoreSalesReport(SalesReport):
    '''In-store report — adds a physical store location.'''
    def __init__(self, region, store_location):
        super().__init__(region)
        self.store_location = store_location

    def print_report(self):
        print(f'Location: {self.store_location}')
        super().print_report()         # reuse parent's print_report logic


class OnlineSalesReport(SalesReport):
    '''Online report — also tracks ship-to states.'''
    def __init__(self, region):
        super().__init__(region)
        self.ship_to_states = []

    def add_record(self, record, state):
        super().add_record(record)     # reuse parent's add_record logic
        self.ship_to_states.append(state)

    def top_state(self):
        return max(set(self.ship_to_states), key=self.ship_to_states.count)

    def print_report(self):
        super().print_report()
        print(f'Top Ship-to State: {self.top_state()}')


# Each child class inherits print_report() and total_sales() from SalesReport
east = StoreSalesReport('East', 'Hartford, CT')
east.add_record(r1)
east.add_record(r2)
east.print_report()

online = OnlineSalesReport('West')
online.add_record(SalesRecord(None, 'Rain Jacket', 89.99), 'CO')
online.add_record(SalesRecord(None, 'Tent', 349.99), 'CO')
online.add_record(SalesRecord(None, 'Hiking Boots', 129.99), 'CA')
online.print_report()