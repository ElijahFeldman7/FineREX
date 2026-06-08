# Dataset Labelling Guidelines
Since our dataset involves licensing restrictions, we are unfortunately unable to provide the source sentences, or outputs based on these cases. However, to ensure future reproducibility of similar experiments, we provide guidelines for dataset annotation.

We use a selection of 1-3 sentence chunks, meant to mimic the extraction at runtime. These are arbitrarily selected, prioritizing the use of sentences that have relavant entities and relationships.

Study some of the example extractions before annotating yourself.

Although these cases contain plentiful legal jargon, they are not the subject of extraction. As esuch, do not extract entities corresponding to governmental organizations or entities closely related to the trial, criminal law and law procedures, such as jury, government, law_enforcement, homeland_security, court, district court, juror, verdict, jury's verdict, hearing, proof of evidence, prosecution, supreme court, federal law, state law, public record, closing argument, greater offense, etc. We are not interested in such Government-related entities.

## Entity type definitions
Below are the entity type definitions. Extract only entities that explicitly match them. Do NOT infer or create new entity types. If a term does not fit any defined entity type, do NOT extract it. Not all entity types will appear in every input chunk, so do NOT misclassify entities.
1. PERSON: Short name or full name of a person from any geographic regions. Smugglers, undocumented non citizens, border patrol agents, etc. are also examples of a PERSON entity.
2. LOCATION: Name of any geographical location, like cities, countries, counties, states, continents, districts, etc. 
3. ORGANIZATION: Names of companies, organized criminal groups, drug cartels, smuggling rings, etc.
4. MEANS_OF_TRANSPORTATION: The mean by which someone moves from one place to another, like car, truck, 18-wheeler, etc.
5. MEANS_OF_COMMUNICATION: The mean by which communication is performed, like phone, WhatsApp, etc.
6. ROUTES: Names of roads, freeways, highways, or other types of roads.
7. SMUGGLED_ITEMS: Any illegally transported goods involved in smuggling activities. This includes drugs, weapons, and other contraband. 

Note: Although people may be transported in the smuggling process, these are humans rather than items. Therefore, they should be extracted as type PERSON entities.

## Extraction Steps
1. Extract entities only if they are explicitly written in the input document without inference or completion. For each extracted entity, extract the following information:
- entity_name: Name of the entity, capitalized.  Do not alter spellings or make corrections. The name should match exactly as written. For example, if 'Jaquez' is extracted as an entity then keep 'Jaquez'. Do not correct it to 'Jacquez'.  
- entity_type: One of the following types: PERSON, LOCATION, MEANS_OF_TRANSPORTATION, MEANS_OF_COMMUNICATION, ROUTES, SMUGGLED_ITEMS, ORGANIZATION
- entity_description: Comprehensive description of the entitys attributes and activities

To reiterate, do not extract any entities related to government organizations or legal proceedings, such as court, jury, government, law enforcement, prosecution, homeland security, etc. These are out of scope and must be excluded if extracted entirely.

Extract each entity type separately in the following order:
- PERSON: Extract all PERSON entities. Title Handling for PERSON entities: If a person’s name appears with a title (e.g., "Border Patrol Agent Bafford Sallee", "Agent Rodriguez", or "Officer David"), extract only the person’s full name (e.g., "Bafford Sallee") as the entity_name. The title (e.g., "Border Patrol Agent") must be included in the entity_description, not in the entity_name. This prevents duplicate nodes and ensures consistent representation of individuals in the knowledge graph.
- LOCATION: Extract all LOCATION entities. If a city and a state appear together (e.g., 'Laredo, Texas' or 'Tucson, Arizona'), treat them as one LOCATION entity in the format 'City, Full State Name'. Do not split them into separate LOCATION entities.
- MEANS_OF_TRANSPORTATION: Extract all MEANS_OF_TRANSPORTATION entities.
- MEANS_OF_COMMUNICATION: Extract all MEANS_OF_COMMUNICATION entities.
- ROUTES: Extract all ROUTES entities.
- SMUGGLED_ITEMS: Extract all SMUGGLED_ITEMS entities
- ORGANIZATION: Extract all ORGANIZATION entities

Format each entity as ("entity"|<entity_name>|<entity_type>|<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are clearly related to each other. Extract all relationships stated explicitly in the input text, even if indirect or embedded in complex structures.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: A numeric score between 0 and 10 indicating the strength of the relationship, based on the following criteria. 0 to 3 (Weak): The relationship is mentioned indirectly, with minimal context. Sentences containing "may have...", "allegedly...", or other uncertain phrasing fall into this category. 4 to 6 (Moderate): The relationship is explicitly stated but lacks detailed context, supporting evidence, or additional information. If the sentence expresses uncertainty but does not use "may have" or "allegedly," it may still fall into this range. 7 to 10 (Strong): The relationship is explicitly stated with clear, detailed context, repeated mentions, or strong supporting evidence. Sentences using direct verb tenses (e.g., "did", "was", "used", "transported") without hedging terms should be rated in this range.

Format each relationship as ("relationship"|<source_entity>|<target_entity>|<relationship_description>|<relationship_strength>)

3. If any government-related entities or relationships were mistakenly extracted (e.g., court, jury, government, prosecution, law enforcement, etc.), remove them. These are out of scope for this task.

4. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **** as the list delimiter.
 
5. When finished, output <END>

######################
-Examples-
Below are three structured examples illustrating entity and relationship extraction. Each example consists of entity types, input text, and the correct output format. Use these examples to learn the correct extraction process.
######################
Example 01:
Input_text:
On March 12, 2024, Sai Deshpande, a known smuggler, drove an 18-wheeler carrying undocumented migrants.
######################
Output:
("entity"|SAI DESHPANDE|PERSON|A known smuggler responsible for transporting migrants in an 18-wheeler)

("entity"|SMUGGLER|PERSON|An individual engaged in illegal human smuggling activities)

("entity"|18-WHEELER|MEANS_OF_TRANSPORTATION|A large truck used for smuggling operations)

("relationship"|SAI DESHPANDE|SMUGGLER|Sai Deshpande is identified as a smuggler involved in this case|8)

("relationship"|SAI DESHPANDE|18-WHEELER|Sai Deshpande drove the 18-wheeler carrying undocumented migrants|9)

<END>


######################
Example 02:
Input_text:
Smugglers from the Horizon Smuggling Ring used remote desert roads to avoid law enforcement, communicating via WhatsApp. The District Court later issued an order against the smuggling ring, and the Government launched an investigation.
######################
Output:
("entity"|SMUGGLERS|PERSON|Individuals engaged in illegal human smuggling activities)

("entity"|HORIZON SMUGGLING RING|ORGANIZATION|An organized smuggling group involved in human trafficking and illegal transportation activities)

("entity"|REMOTE DESERT ROADS|ROUTES|A smuggling route used to move migrants undetected)

("entity"|WHATSAPP|MEANS_OF_COMMUNICATION|Application used by smugglers to coordinate and evade law enforcement)

("relationship"|SMUGGLERS|HORIZON SMUGGLING RING|The smugglers were associated with the Horizon Smuggling Ring|7)

("relationship"|SMUGGLERS|REMOTE DESERT ROADS|Smugglers used this route to avoid law enforcement|8)

("relationship"|SMUGGLERS|WHATSAPP|Smugglers used WhatsApp to coordinate while avoiding detection|7)

("relationship"|HORIZON SMUGGLING RING|REMOTE DESERT ROADS|The Horizon Smuggling Ring used this route for illegal transportation|7)

<END>


######################
Example 03:
Input_text:
Krish Patil transported undocumented aliens along Interstate 988 before arriving at a stash house in Velu, Gujarat where illegal weapons were stored.
######################
Output:
("entity"|KRISH PATIL|PERSON|A smuggler involved in transporting undocumented aliens and illegal weapons)

("entity"|SMUGGLER|PERSON|An individual engaged in illegal human smuggling activities)

("entity"|UNDOCUMENTED ALIENS|PERSON|A group of individuals smuggled across the border without legal documentation)

("entity"|ILLEGAL WEAPONS|SMUGGLED_ITEMS|Firearms and other restricted weapons illegally transported and stored)

("entity"|INTERSTATE 988|ROUTES|A known smuggling route used to transport undocumented aliens without detection)

("entity"|VELU, GUJARAT|LOCATION|A city where illegal weapons were stored and smuggling operations were coordinated)

("entity"|STASH HOUSE|LOCATION|A hidden facility used to shelter undocumented aliens and store illegal weapons before further transport)

("relationship"|KRISH PATIL|SMUGGLER|Krish Patil is identified as a smuggler involved in this case|9)

("relationship"|KRISH PATIL|UNDOCUMENTED ALIENS|Krish Patil was responsible for smuggling undocumented aliens along Interstate 988|10)

("relationship"|KRISH PATIL|ILLEGAL WEAPONS|Krish Patil was involved in smuggling and storing illegal weapons at the stash house|9)

("relationship"|UNDOCUMENTED ALIENS|INTERSTATE 988|Undocumented aliens were transported via Interstate 988 to avoid detection|9)

("relationship"|ILLEGAL WEAPONS|STASH HOUSE|Illegal weapons were stored in the stash house before being distributed|9)

("relationship"|UNDOCUMENTED ALIENS|STASH HOUSE|Undocumented aliens were brought to the stash house before further transport|8)

("relationship"|STASH HOUSE|VELU, GUJARAT|The stash house was located in Velu, Gujarat serving as a hub for illegal activities|8)

<END>