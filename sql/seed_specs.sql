-- Seed a pilot product type and default thickness specs

IF NOT EXISTS (SELECT 1 FROM dbo.product_type WHERE type_code='Type_1')
BEGIN
  INSERT INTO dbo.product_type(type_code, description)
  VALUES ('Type_1', 'Pilot product');
END

DECLARE @type_id INT = (SELECT type_id FROM dbo.product_type WHERE type_code='Type_1');

MERGE dbo.thickness_spec AS t
USING (VALUES
('A',100,160,130,1.0),
('B',120,180,150,1.2),
('C',120,180,150,1.2),
('D',100,180,150,1.0),
('E',100,180,140,1.0),
('F',100,180,140,1.0),
('G',100,180,140,1.0)
) AS s(point, lo, hi, tgt, w)
ON t.type_id=@type_id AND t.point=s.point
WHEN MATCHED THEN
  UPDATE SET lower_um=s.lo, upper_um=s.hi, default_target_um=s.tgt, weight=s.w
WHEN NOT MATCHED THEN
  INSERT(type_id, point, lower_um, upper_um, default_target_um, weight)
  VALUES(@type_id, s.point, s.lo, s.hi, s.tgt, s.w);