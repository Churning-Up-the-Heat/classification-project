def online_features(train, test, validate):
    train['online_features']    = train.online_security_encoded + train.device_protection_encoded + train.tech_support_encoded
    validate['online_features'] = validate.online_security_encoded + validate.device_protection_encoded + validate.tech_support_encoded
    test['online_features']     = test.online_security_encoded + test.device_protection_encoded + test.tech_support_encoded
    
    return train, test, validate