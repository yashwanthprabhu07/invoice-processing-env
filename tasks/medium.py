import random

VENDORS = [
    "Nexus Traders", "Orbit Supplies Co.", "Greenfield Exports",
    "BlueWave Tech", "Crestline Partners", "Horizon Goods Ltd",
    "Sterling Works", "Pinnacle Trade Co.", "Redwood Ventures", "Axon Supplies",
]

AMOUNTS = [
    "2360", "4750", "6100", "8800", "5500",
    "3900", "7200", "9100", "11000", "4200",
]

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

NOISE_LINES = [
    "Payment terms: Net 30",
    "Payment terms: Net 45",
    "GST included",
    "Tax (18%): included",
    "Please pay within 15 days",
    "Due in 45 days from invoice date",
    "Contact: billing@vendor.com",
    "Ref No: INV-{ref}",
    "PO Number: PO-{ref}",
    "Late fee applies after due date",
]

def get_medium_task():
    vendor = random.choice(VENDORS)
    amount = random.choice(AMOUNTS)
    day = random.randint(1, 28)
    month = random.choice(MONTHS)
    ref = random.randint(1000, 9999)

    noise = random.sample(NOISE_LINES, k=2)
    noise = [line.replace("{ref}", str(ref)) for line in noise]

    invoice_text = (
        f"INVOICE #{ref}\n"
        f"Issued by: {vendor}\n"
        f"Billing Date: {day} {month} 2024\n"
        f"Total Amount Due: ${amount}\n"
        f"{noise[0]}\n"
        f"{noise[1]}\n"
    )
    return {
        "invoice_text": invoice_text,
        "true_fields": {
            "amount": amount,
            "vendor": vendor,
            "date": f"{day} {month}",
        },
        "fraud": False,
    }