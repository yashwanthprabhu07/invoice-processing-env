import random

VENDORS = [
    "FastTrack Logistics", "Bright Horizons Ltd", "Meridian Supplies",
    "Apex Global Traders", "SilverLine Contractors", "Quantum Freight Co.",
    "Redline Partners", "Vortex Supplies", "Caspian Works", "Nimbus Traders",
]

AMOUNTS = [
    "15000", "19500", "11000", "50000", "12800",
    "23000", "8500", "31000", "44000", "17500",
]

MONTHS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]

FRAUD_SIGNALS = [
    "NOTE: Duplicate of a previous invoice — resubmitted for processing",
    "Bank: Different account than registered",
    "Discrepancy note: Rate revised without approval",
    "Warning: Vendor name does not match registered records",
    "Payment urgently requested within 24 hours",
    "Note: Third invoice this month for same items",
    "Amount billed is 4x the approved quote",
    "No supporting documents attached",
    "Previous invoices already paid — resubmission flagged",
    "Urgent: Account details changed — update before payment",
]

def get_hard_task():
    vendor = random.choice(VENDORS)
    amount = random.choice(AMOUNTS)
    day = random.randint(1, 28)
    month = random.choice(MONTHS)
    ref = random.randint(1000, 9999)

    signals = random.sample(FRAUD_SIGNALS, k=2)

    invoice_text = (
        f"INVOICE #{ref}\n"
        f"Vendor: {vendor}\n"
        f"Date: {day} {month} 2024\n"
        f"Amount Due: ${amount}\n"
        f"{signals[0]}\n"
        f"{signals[1]}\n"
    )
    return {
        "invoice_text": invoice_text,
        "true_fields": {
            "amount": amount,
            "vendor": vendor,
            "date": f"{day} {month}",
        },
        "fraud": True,
    }