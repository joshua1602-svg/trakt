from lxml import etree
import os
import sys

def validate(xml_path, xsd_path):
    print(f"Validating: {xml_path}")
    print(f"Against:    {xsd_path}")
    print("-" * 60)

    if not os.path.exists(xml_path):
        print("❌ Error: XML file not found.")
        return
    if not os.path.exists(xsd_path):
        print("❌ Error: XSD file not found.")
        return

    try:
        # Load XSD
        xmlschema_doc = etree.parse(xsd_path)
        xmlschema = etree.XMLSchema(xmlschema_doc)
        
        # Load XML
        doc = etree.parse(xml_path)
        
        # Validate
        if xmlschema.validate(doc):
            print("✅ SUCCESS! The XML is valid.")
        else:
            print("❌ FAILED. Errors found:")
            for i, error in enumerate(xmlschema.error_log):
                print(f"  [Line {error.line}] {error.message}")
                if i >= 20: 
                    print("  ... (stopping after 20 errors)")
                    break
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {e}")

if __name__ == "__main__":
    # Hardcoded filenames based on your previous messages
    XML_FILE = "annex2_report_template.xml"
    XSD_FILE = "DRAFT1auth.099.001.04_1.3.0.xsd"
    
    validate(XML_FILE, XSD_FILE)